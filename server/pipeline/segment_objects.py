"""Object segmentation using SAM2 — post-hoc approach.

Runs SAM2 on rendered/input frames, then backprojects 2D masks to 3D gaussians
using camera poses from AnySplat or COLMAP.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SAM2_DIR = Path("/home/nock/projects/sam2")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_small.pt"

# Lazy-loaded singleton
_sam2_model = None
_mask_generator = None
_image_predictor = None


def _ensure_sam2_path():
    """Add SAM2 to sys.path if not already there."""
    sam2_str = str(SAM2_DIR)
    if sam2_str not in sys.path:
        sys.path.insert(0, sam2_str)


def _get_mask_generator():
    """Lazy-load SAM2 auto mask generator (singleton)."""
    global _sam2_model, _mask_generator
    if _mask_generator is not None:
        return _mask_generator

    import torch
    _ensure_sam2_path()
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SAM2 on {device}...")
    _sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    _mask_generator = SAM2AutomaticMaskGenerator(
        _sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=500,
    )
    logger.info("SAM2 auto mask generator ready")
    return _mask_generator


def _get_image_predictor():
    """Lazy-load SAM2 image predictor for click-based segmentation."""
    global _sam2_model, _image_predictor
    if _image_predictor is not None:
        return _image_predictor

    import torch
    _ensure_sam2_path()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if _sam2_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SAM2 on {device}...")
        _sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)

    _image_predictor = SAM2ImagePredictor(_sam2_model)
    logger.info("SAM2 image predictor ready")
    return _image_predictor


def segment_frame_auto(image_path: Path) -> list[dict]:
    """Run SAM2 automatic mask generation on a single frame.

    Returns list of segments, each with:
      - segmentation: bool mask as RLE (run-length encoded for storage)
      - area: pixel area
      - bbox: [x, y, w, h]
      - predicted_iou: confidence
      - stability_score: mask quality
    """
    mask_gen = _get_mask_generator()
    img = np.array(Image.open(image_path).convert("RGB"))
    masks = mask_gen.generate(img)

    # Sort by area (largest first) and convert masks for JSON storage
    masks.sort(key=lambda m: m["area"], reverse=True)

    results = []
    for i, m in enumerate(masks):
        results.append({
            "id": i,
            "area": int(m["area"]),
            "bbox": [int(x) for x in m["bbox"]],
            "predicted_iou": round(float(m["predicted_iou"]), 4),
            "stability_score": round(float(m["stability_score"]), 4),
        })
    return results, masks


def segment_frame_click(image_path: Path, point_x: int, point_y: int) -> dict:
    """Segment the object at a click point using SAM2 image predictor.

    Returns the best mask for the clicked point.
    """
    predictor = _get_image_predictor()
    img = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(img)

    point_coords = np.array([[point_x, point_y]])
    point_labels = np.array([1])  # foreground

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Pick best mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    return {
        "mask": mask,  # bool numpy array
        "score": float(scores[best_idx]),
        "area": int(mask.sum()),
    }


def auto_segment_project(
    frames_dir: Path,
    project_dir: Path,
    max_frames: int = 5,
) -> dict:
    """Run automatic segmentation on key frames and merge into 3D segments.

    Steps:
    1. Pick a few representative frames (spread across the sequence)
    2. Run SAM2 auto mask generation on each
    3. Save per-frame masks as PNGs
    4. Write segment_manifest.json

    Returns the segment manifest.
    """
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return {"segments": [], "total": 0}

    # Pick spread-out frames
    n = min(max_frames, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    selected_frames = [frames[i] for i in indices]

    segments_dir = project_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    all_frame_segments = []

    for frame_path in selected_frames:
        logger.info(f"Segmenting {frame_path.name}...")
        results, raw_masks = segment_frame_auto(frame_path)

        # Save colored mask overlay as PNG
        if raw_masks:
            img = np.array(Image.open(frame_path).convert("RGB"))
            overlay = img.copy().astype(np.float32)

            # Generate distinct colors for each segment
            colors = _generate_colors(len(raw_masks))

            for i, m in enumerate(raw_masks):
                mask_bool = m["segmentation"]
                color = colors[i]
                overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5

            overlay_img = Image.fromarray(overlay.astype(np.uint8))
            overlay_path = segments_dir / f"{frame_path.stem}_segments.jpg"
            overlay_img.save(str(overlay_path), quality=85)

            # Save individual masks as compact numpy arrays
            for i, m in enumerate(raw_masks):
                mask_path = segments_dir / f"{frame_path.stem}_mask_{i:03d}.npy"
                np.save(str(mask_path), m["segmentation"].astype(np.uint8))

        all_frame_segments.append({
            "frame": frame_path.name,
            "n_segments": len(results),
            "segments": results,
        })

    # Build segment manifest
    # Use the frame with the most segments as the "primary" view
    primary = max(all_frame_segments, key=lambda f: f["n_segments"])

    segments = []
    for seg in primary["segments"]:
        segments.append({
            "id": seg["id"],
            "label": f"object_{seg['id']}",
            "area": seg["area"],
            "bbox": seg["bbox"],
            "confidence": seg["predicted_iou"],
            "primary_frame": primary["frame"],
            "color": _generate_colors(len(primary["segments"]))[seg["id"]],
            "gaussian_ids": [],  # Filled by assign_gaussians_to_segments
        })

    manifest = {
        "segments": segments,
        "total": len(segments),
        "per_frame": all_frame_segments,
        "primary_frame": primary["frame"],
    }

    manifest_path = project_dir / "segment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"Segmentation complete: {len(segments)} objects found")
    return manifest


def click_segment_project(
    frames_dir: Path,
    project_dir: Path,
    frame_filename: str,
    point_x: int,
    point_y: int,
) -> dict:
    """Segment a specific object via click point and add to manifest."""
    frame_path = frames_dir / frame_filename
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    result = segment_frame_click(frame_path, point_x, point_y)

    segments_dir = project_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    # Load or create manifest
    manifest_path = project_dir / "segment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"segments": [], "total": 0, "per_frame": [], "primary_frame": frame_filename}

    # Create new segment
    seg_id = len(manifest["segments"])
    color = _generate_colors(seg_id + 1)[seg_id]

    # Save mask
    mask_path = segments_dir / f"click_mask_{seg_id:03d}.npy"
    np.save(str(mask_path), result["mask"].astype(np.uint8))

    # Save overlay for this mask
    img = np.array(Image.open(frame_path).convert("RGB"))
    overlay = img.copy().astype(np.float32)
    overlay[result["mask"]] = overlay[result["mask"]] * 0.4 + np.array(color) * 0.6
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_path = segments_dir / f"click_segment_{seg_id:03d}.jpg"
    overlay_img.save(str(overlay_path), quality=85)

    new_segment = {
        "id": seg_id,
        "label": f"object_{seg_id}",
        "area": result["area"],
        "bbox": _mask_to_bbox(result["mask"]),
        "confidence": result["score"],
        "primary_frame": frame_filename,
        "color": color,
        "gaussian_ids": [],
        "click_point": [point_x, point_y],
    }

    manifest["segments"].append(new_segment)
    manifest["total"] = len(manifest["segments"])
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return new_segment


def assign_gaussians_to_segments(
    project_dir: Path,
    frames_dir: Path,
) -> dict:
    """Backproject 2D masks to 3D gaussians using camera poses.

    Reads cameras.json (from AnySplat) or COLMAP cameras, projects each
    gaussian center into each frame, and assigns it to the segment whose
    mask it falls into (multi-view voting).
    """
    manifest_path = project_dir / "segment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No segment manifest. Run segmentation first.")

    manifest = json.loads(manifest_path.read_text())
    ply_path = project_dir / "scene.ply"
    cameras_path = project_dir / "cameras.json"

    if not ply_path.exists():
        raise FileNotFoundError("No scene.ply found")

    # Load gaussian positions
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]
    xyz = np.column_stack([
        np.array(vertices["x"]),
        np.array(vertices["y"]),
        np.array(vertices["z"]),
    ])
    n_gaussians = len(xyz)
    logger.info(f"Loaded {n_gaussians} gaussians from {ply_path}")

    # Load cameras
    if cameras_path.exists():
        cam_data = json.loads(cameras_path.read_text())
        cameras = cam_data["cameras"]
        resolution = cam_data.get("resolution", 448)
        focal = cam_data.get("focal_length_px", resolution * 0.85)
    else:
        logger.warning("No cameras.json — gaussian assignment requires camera poses")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest

    # For each segment, collect masks from primary frame
    segments_dir = project_dir / "segments"
    segment_masks = {}

    for seg in manifest["segments"]:
        # Try click mask first, then auto mask
        click_mask = segments_dir / f"click_mask_{seg['id']:03d}.npy"
        auto_mask = segments_dir / f"{Path(seg['primary_frame']).stem}_mask_{seg['id']:03d}.npy"

        if click_mask.exists():
            segment_masks[seg["id"]] = np.load(str(click_mask)).astype(bool)
        elif auto_mask.exists():
            segment_masks[seg["id"]] = np.load(str(auto_mask)).astype(bool)

    if not segment_masks:
        logger.warning("No masks found for any segment")
        return manifest

    # Find the camera for the primary frame
    primary_frame = manifest.get("primary_frame", "")
    primary_cam = None
    for cam in cameras:
        if cam["frame"] == primary_frame or cam["frame"] in primary_frame:
            primary_cam = cam
            break

    if primary_cam is None and cameras:
        # Use first camera as fallback
        primary_cam = cameras[0]

    if primary_cam is None:
        logger.warning("No matching camera found for primary frame")
        return manifest

    # Project gaussians into the primary frame
    transform = np.array(primary_cam["transform"])  # [3, 4] or [4, 4]
    if transform.shape[0] == 4:
        transform = transform[:3]  # Take top 3 rows

    R = transform[:, :3]  # [3, 3]
    t = transform[:, 3]   # [3]

    # Camera-to-world → we need world-to-camera
    R_inv = R.T
    t_inv = -R_inv @ t

    # Project: p_cam = R_inv @ p_world + t_inv
    xyz_cam = (R_inv @ xyz.T).T + t_inv  # [N, 3]

    # Perspective projection
    cx = resolution / 2
    cy = resolution / 2
    fx = fy = focal

    # Only consider points in front of camera
    valid = xyz_cam[:, 2] > 0.01
    u = np.full(n_gaussians, -1.0)
    v = np.full(n_gaussians, -1.0)

    u[valid] = fx * xyz_cam[valid, 0] / xyz_cam[valid, 2] + cx
    v[valid] = fy * xyz_cam[valid, 1] / xyz_cam[valid, 2] + cy

    # Assign each gaussian to a segment
    gaussian_segments = np.full(n_gaussians, -1, dtype=np.int32)

    for seg_id, mask in segment_masks.items():
        h, w = mask.shape
        in_bounds = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui = np.clip(u.astype(int), 0, w - 1)
        vi = np.clip(v.astype(int), 0, h - 1)

        hits = in_bounds & mask[vi, ui]
        gaussian_segments[hits] = seg_id

    # Update manifest with gaussian assignments
    for seg in manifest["segments"]:
        seg_mask = gaussian_segments == seg["id"]
        seg["gaussian_ids"] = np.where(seg_mask)[0].tolist()
        seg["n_gaussians"] = int(seg_mask.sum())
        logger.info(f"Segment '{seg['label']}': {seg['n_gaussians']} gaussians")

    # Count unassigned
    unassigned = int((gaussian_segments == -1).sum())
    manifest["unassigned_gaussians"] = unassigned
    manifest["total_gaussians"] = n_gaussians
    logger.info(f"Assigned: {n_gaussians - unassigned}/{n_gaussians} ({unassigned} unassigned)")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Convert bool mask to [x, y, w, h] bbox."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def _generate_colors(n: int) -> list[list[int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360  # Golden angle
        # HSV to RGB (simplified)
        h = hue / 60
        c = 200  # Chroma
        x = int(c * (1 - abs(h % 2 - 1)))
        c = int(c)
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.append([r + 55, g + 55, b + 55])  # Boost brightness
    return colors
