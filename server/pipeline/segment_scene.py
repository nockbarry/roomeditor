"""Scene segmentation pipeline using SAM2 with multi-view Gaussian voting.

Runs SAM2 auto mask generation on multiple views, then assigns each Gaussian
to a segment via majority voting across all frames where it's visible.
"""

import json
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def project_gaussians_to_frame(
    means: np.ndarray,
    cam: dict,
    resolution: int,
    focal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Gaussian centers into a camera frame.

    Args:
        means: (N, 3) Gaussian positions in world space
        cam: Camera dict with 'transform' key (camera-to-world [3,4] or [4,4])
        resolution: Image resolution (width = height)
        focal: Focal length in pixels

    Returns:
        u: (N,) pixel x coordinates
        v: (N,) pixel y coordinates
        in_bounds: (N,) boolean mask of valid projections
    """
    n = len(means)
    transform = np.array(cam["transform"], dtype=np.float64)
    if transform.shape[0] == 4:
        transform = transform[:3]

    R = transform[:, :3]
    t = transform[:, 3]

    # Camera-to-world -> world-to-camera
    R_inv = R.T
    t_inv = -R_inv @ t

    xyz_cam = (R_inv @ means.T).T + t_inv

    cx = resolution / 2
    cy = resolution / 2
    fx = fy = focal

    valid = xyz_cam[:, 2] > 0.01
    u = np.full(n, -1.0)
    v = np.full(n, -1.0)

    u[valid] = fx * xyz_cam[valid, 0] / xyz_cam[valid, 2] + cx
    v[valid] = fy * xyz_cam[valid, 1] / xyz_cam[valid, 2] + cy

    in_bounds = valid & (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)

    return u, v, in_bounds


async def segment_scene(
    ply_path: Path,
    images_dir: Path,
    output_seg_path: Path,
    cameras_path: Path | None = None,
    max_frames: int = 5,
    progress_callback=None,
) -> list[dict]:
    """Segment a trained Gaussian scene into objects using multi-view voting.

    Full pipeline:
    1. Select max_frames evenly-spaced frames
    2. Run SAM2 auto mask generation on each
    3. Multi-view Gaussian assignment via vote matrix
    4. Write segment manifest and index map

    Returns a list of segment metadata dicts.
    """
    from plyfile import PlyData
    from pipeline.segment_objects import segment_frame_auto, _generate_colors

    # Load Gaussian positions
    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]
    xyz = np.column_stack([
        np.array(vertices["x"]),
        np.array(vertices["y"]),
        np.array(vertices["z"]),
    ])
    n_gaussians = len(xyz)
    logger.info(f"Loaded {n_gaussians} gaussians for segmentation")

    # Load cameras
    if cameras_path and cameras_path.exists():
        cam_data = json.loads(cameras_path.read_text())
        cameras = cam_data["cameras"]
        resolution = cam_data.get("resolution", 448)
        focal = cam_data.get("focal_length_px", resolution * 0.85)
    else:
        logger.warning("No cameras.json for multi-view segmentation, falling back to stub")
        return _stub_segment(n_gaussians, output_seg_path)

    # Select evenly-spaced frames
    frames = sorted(images_dir.glob("frame_*.jpg"))
    if not frames:
        return _stub_segment(n_gaussians, output_seg_path)

    n = min(max_frames, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    selected_frames = [frames[i] for i in indices]

    if progress_callback:
        await progress_callback(0.05, "Running SAM2 on selected frames")

    # Step 1: Run SAM2 on each selected frame
    all_frame_masks = []  # list of (frame_path, raw_masks, cam)
    for fi, frame_path in enumerate(selected_frames):
        logger.info(f"Segmenting {frame_path.name} ({fi+1}/{n})...")
        results, raw_masks = segment_frame_auto(frame_path)

        # Find matching camera for this frame
        matching_cam = None
        for cam in cameras:
            if cam["frame"] == frame_path.name or frame_path.name in cam.get("frame", ""):
                matching_cam = cam
                break

        if matching_cam is None:
            # Try matching by index
            frame_idx = int(frame_path.stem.split("_")[-1])
            if frame_idx < len(cameras):
                matching_cam = cameras[frame_idx]

        if matching_cam is not None and raw_masks:
            all_frame_masks.append((frame_path, raw_masks, matching_cam))

        if progress_callback:
            await progress_callback(0.05 + 0.45 * (fi + 1) / n, f"Segmented {fi+1}/{n} frames")

    if not all_frame_masks:
        logger.warning("No frames with valid cameras and masks")
        return _stub_segment(n_gaussians, output_seg_path)

    # Step 2: Determine global segment count from the primary (most-segments) frame
    primary_idx = max(range(len(all_frame_masks)), key=lambda i: len(all_frame_masks[i][1]))
    primary_frame_path, primary_masks, _ = all_frame_masks[primary_idx]
    n_segments = len(primary_masks)

    if progress_callback:
        await progress_callback(0.55, f"Assigning {n_gaussians} gaussians to {n_segments} segments")

    # Step 3: Multi-view vote matrix (n_gaussians, n_segments)
    vote_matrix = np.zeros((n_gaussians, n_segments), dtype=np.float32)
    visibility_count = np.zeros(n_gaussians, dtype=np.int32)

    for fi, (frame_path, raw_masks, cam) in enumerate(all_frame_masks):
        u, v, in_bounds = project_gaussians_to_frame(xyz, cam, resolution, focal)

        # Count visibility
        visibility_count[in_bounds] += 1

        # For primary frame: use primary masks directly
        # For other frames: use masks from that frame (may differ in count)
        masks_to_use = raw_masks if fi == primary_idx else raw_masks

        for seg_idx in range(min(len(masks_to_use), n_segments)):
            mask = masks_to_use[seg_idx]["segmentation"]
            h, w = mask.shape

            # For each Gaussian in bounds, check if it falls in this mask
            ui = np.clip(u[in_bounds].astype(int), 0, w - 1)
            vi = np.clip(v[in_bounds].astype(int), 0, h - 1)

            hits = mask[vi, ui]
            # Weighted by IoU confidence
            confidence = masks_to_use[seg_idx].get("predicted_iou", 0.8)
            vote_matrix[in_bounds, seg_idx] += hits.astype(np.float32) * confidence

        if progress_callback:
            await progress_callback(
                0.55 + 0.35 * (fi + 1) / len(all_frame_masks),
                f"Projected frame {fi+1}/{len(all_frame_masks)}"
            )

    # Step 4: Assign by majority vote
    gaussian_segments = np.full(n_gaussians, -1, dtype=np.int32)

    # Gaussians with votes
    has_votes = vote_matrix.max(axis=1) > 0
    gaussian_segments[has_votes] = vote_matrix[has_votes].argmax(axis=1)

    # Gaussians visible in < 2 frames with no majority: use highest single-view confidence
    low_vis = (visibility_count < 2) & has_votes
    gaussian_segments[low_vis] = vote_matrix[low_vis].argmax(axis=1)

    if progress_callback:
        await progress_callback(0.92, "Writing segment manifest")

    # Step 5: Build segment manifest
    colors = _generate_colors(n_segments)
    segments_dir = output_seg_path.parent / "segments"
    segments_dir.mkdir(exist_ok=True)

    segments = []
    for seg_idx in range(n_segments):
        seg_mask = gaussian_segments == seg_idx
        gids = np.where(seg_mask)[0].tolist()
        n_gs = int(seg_mask.sum())

        # Compute centroid from assigned gaussians
        if n_gs > 0:
            centroid = xyz[seg_mask].mean(axis=0).tolist()
            bbox_min = xyz[seg_mask].min(axis=0).tolist()
            bbox_max = xyz[seg_mask].max(axis=0).tolist()
        else:
            centroid = [0.0, 0.0, 0.0]
            bbox_min = [0.0, 0.0, 0.0]
            bbox_max = [0.0, 0.0, 0.0]

        # Save mask from primary frame
        if seg_idx < len(primary_masks):
            mask_path = segments_dir / f"{primary_frame_path.stem}_mask_{seg_idx:03d}.npy"
            np.save(str(mask_path), primary_masks[seg_idx]["segmentation"].astype(np.uint8))

        segments.append({
            "id": seg_idx,
            "label": f"object_{seg_idx}",
            "area": int(primary_masks[seg_idx]["area"]) if seg_idx < len(primary_masks) else 0,
            "bbox": [int(x) for x in primary_masks[seg_idx]["bbox"]] if seg_idx < len(primary_masks) else [0, 0, 0, 0],
            "confidence": round(float(primary_masks[seg_idx].get("predicted_iou", 0.8)), 4) if seg_idx < len(primary_masks) else 0.0,
            "primary_frame": primary_frame_path.name,
            "color": colors[seg_idx],
            "gaussian_ids": gids,
            "n_gaussians": n_gs,
            "centroid": centroid,
            "visible": True,
        })

    unassigned = int((gaussian_segments == -1).sum())
    manifest = {
        "segments": segments,
        "total": len(segments),
        "primary_frame": primary_frame_path.name,
        "total_gaussians": n_gaussians,
        "unassigned_gaussians": unassigned,
        "per_frame": [{"frame": fp.name, "n_segments": len(ms)} for fp, ms, _ in all_frame_masks],
    }

    # Write manifest
    manifest_path = output_seg_path.parent / "segment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write segment index map (Uint8Array)
    index_map = np.zeros(n_gaussians, dtype=np.uint8)
    for seg_idx, seg in enumerate(segments):
        if seg_idx + 1 > 255:
            break
        for gid in seg["gaussian_ids"]:
            if 0 <= gid < n_gaussians:
                index_map[gid] = seg_idx + 1
    index_map.tofile(output_seg_path)

    if progress_callback:
        await progress_callback(1.0, "Segmentation complete")

    assigned = n_gaussians - unassigned
    logger.info(f"Multi-view segmentation: {len(segments)} segments, {assigned}/{n_gaussians} assigned ({unassigned} unassigned)")

    return segments


def _stub_segment(n_gaussians: int, output_seg_path: Path) -> list[dict]:
    """Fallback: single segment containing all Gaussians."""
    seg_data = np.zeros(n_gaussians, dtype=np.uint16)
    seg_data.tofile(output_seg_path)
    logger.info(f"Wrote placeholder segment map ({n_gaussians} Gaussians)")
    return [{
        "segment_id": 0,
        "label": "room",
        "gaussian_start": 0,
        "gaussian_count": n_gaussians,
        "centroid": [0.0, 0.0, 0.0],
        "bbox_min": [-10.0, -10.0, -10.0],
        "bbox_max": [10.0, 10.0, 10.0],
    }]


def _count_ply_vertices(ply_path: Path) -> int:
    """Read the vertex count from a PLY file header."""
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise RuntimeError(f"Could not find vertex count in {ply_path}")
