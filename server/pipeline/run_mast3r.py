"""MASt3R-based pose estimation as a drop-in COLMAP replacement.

Uses the MASt3R vision transformer (from InstantSplat installation) to
estimate camera poses and dense 3D point clouds from unposed images.
Outputs the same COLMAP-compatible binary format so the downstream
Gaussian training pipeline works unchanged.

Advantages over COLMAP:
- Works with fewer images (down to 2-3)
- Much faster (seconds vs minutes)
- More robust to phone video quality (motion blur, rolling shutter)
- Produces dense depth maps for better Gaussian initialization
"""

import logging
import struct
import sys
from pathlib import Path
from time import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# InstantSplat directory where MASt3R is installed
INSTANTSPLAT_DIR = Path("/home/nock/projects/instantsplat")
MAST3R_CHECKPOINT = INSTANTSPLAT_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


def _ensure_mast3r_on_path():
    """Add InstantSplat directories to sys.path for MASt3R imports."""
    paths_to_add = [
        str(INSTANTSPLAT_DIR),
        str(INSTANTSPLAT_DIR / "dust3r"),
        str(INSTANTSPLAT_DIR / "croco"),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def run_mast3r(
    images_dir: Path,
    output_dir: Path,
    image_size: int = 512,
    n_views: int = 0,
    conf_aware_ranking: bool = True,
    co_vis_dsp: bool = True,
    depth_threshold: float = 0.01,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict]:
    """Run MASt3R pose estimation pipeline.

    Replaces COLMAP SfM with MASt3R-based geometry estimation.
    Produces COLMAP-compatible binary files in output_dir/sparse/0/.

    Args:
        images_dir: Directory containing input images (jpg/png).
        output_dir: Output directory for COLMAP-format results.
        image_size: Resize images to this size for MASt3R (default 512).
        n_views: Number of views to use (0 = all available images).
        conf_aware_ranking: Use confidence-aware view ranking.
        co_vis_dsp: Use co-visibility aware downsampling.
        depth_threshold: Depth consistency threshold for co-visibility.
        progress_callback: Optional callback for progress updates (0-1).

    Returns:
        (model_dir, metadata) where model_dir is the sparse reconstruction
        directory and metadata includes reconstruction stats.
    """
    import torch

    _ensure_mast3r_on_path()

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Step 1: Load MASt3R model ----------
    logger.info("MASt3R: loading model...")
    from mast3r.model import AsymmetricMASt3R

    model = AsymmetricMASt3R.from_pretrained(str(MAST3R_CHECKPOINT)).to(device)

    if progress_callback:
        progress_callback(0.1)

    # ---------- Step 2: Load and prepare images ----------
    logger.info(f"MASt3R: loading images from {images_dir}")
    image_files = _get_sorted_images(images_dir)

    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    # Subsample if n_views specified
    if n_views > 0 and n_views < len(image_files):
        step = max(1, len(image_files) // n_views)
        image_files = image_files[::step][:n_views]

    actual_n_views = len(image_files)
    logger.info(f"MASt3R: using {actual_n_views} images")

    images, org_shape = _load_images_for_mast3r(image_files, image_size)

    if progress_callback:
        progress_callback(0.2)

    # ---------- Step 3: Run MASt3R inference ----------
    logger.info("MASt3R: computing image pairs...")
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy
    from dust3r.utils.geometry import inv

    t_start = time()
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    if progress_callback:
        progress_callback(0.3)

    logger.info(f"MASt3R: running inference on {len(pairs)} image pairs...")
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    if progress_callback:
        progress_callback(0.5)

    # ---------- Step 4: Global alignment ----------
    logger.info("MASt3R: global alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01, focal_avg=True)

    if progress_callback:
        progress_callback(0.7)

    # ---------- Step 5: Extract scene data ----------
    extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
    intrinsics = to_numpy(scene.get_intrinsics())
    focals = to_numpy(scene.get_focals())
    imgs = np.array(scene.imgs)
    pts3d = to_numpy(scene.get_pts3d())
    depthmaps = to_numpy(scene.get_depthmaps())
    confs_raw = [param.detach().cpu().numpy() for param in scene.im_conf]
    confs = np.array(confs_raw)

    # Confidence-aware ranking
    if conf_aware_ranking and actual_n_views > 2:
        avg_conf = confs.mean(axis=(1, 2))
        sorted_indices = np.argsort(avg_conf)[::-1]
        logger.info(f"MASt3R: confidence ranking: {sorted_indices}")
    else:
        sorted_indices = np.arange(actual_n_views)

    # Co-visibility masking
    overlapping_masks = None
    if co_vis_dsp and depth_threshold > 0:
        logger.info("MASt3R: computing co-visibility masks...")
        overlapping_masks = _compute_co_vis_masks(
            sorted_indices, depthmaps, pts3d, intrinsics,
            extrinsics_w2c, imgs.shape, depth_threshold
        )
        overlapping_masks = ~overlapping_masks

    if progress_callback:
        progress_callback(0.85)

    # ---------- Step 6: Save as COLMAP format ----------
    logger.info("MASt3R: saving COLMAP-compatible output...")
    org_w, org_h = org_shape

    # Use consistent focal for all views
    focal = focals[0]
    scale_x = org_w / imgs.shape[2]
    scale_y = org_h / imgs.shape[1]
    scaled_focal_x = focal * scale_x
    scaled_focal_y = focal * scale_y

    # Save cameras.bin
    _write_cameras_bin(
        sparse_dir / "cameras.bin",
        actual_n_views,
        org_w, org_h,
        scaled_focal_x, scaled_focal_y,
    )

    # Save images.bin
    _write_images_bin(
        sparse_dir / "images.bin",
        extrinsics_w2c,
        image_files,
    )

    # Save points3D as PLY (for Gaussian initialization)
    if overlapping_masks is not None:
        pts_flat = np.concatenate([p[m] for p, m in zip(pts3d, overlapping_masks)])
        col_flat = np.concatenate([p[m] for p, m in zip(imgs, overlapping_masks)])
        conf_flat = np.concatenate([
            c[m.reshape(-1)] for c, m in zip(
                confs.reshape(confs.shape[0], -1), overlapping_masks
            )
        ])
    else:
        pts_flat = pts3d.reshape(-1, 3)
        col_flat = imgs.reshape(-1, 3)
        conf_flat = confs.reshape(-1)

    pts_flat = pts_flat.reshape(-1, 3)
    col_flat = (col_flat.reshape(-1, 3) * 255).astype(np.uint8)

    # Write points3D.bin in COLMAP format
    _write_points3d_bin(sparse_dir / "points3D.bin", pts_flat, col_flat)

    # Also save confidence map for potential use in training
    np.save(sparse_dir / "confidence.npy", confs)
    np.save(sparse_dir / "depth_maps.npy", depthmaps)

    elapsed = time() - t_start

    # Clean up GPU memory
    del model, scene, output
    torch.cuda.empty_cache()

    if progress_callback:
        progress_callback(1.0)

    n_points = len(pts_flat)
    logger.info(
        f"MASt3R reconstruction complete: {actual_n_views} images, "
        f"{n_points} 3D points in {elapsed:.1f}s"
    )

    meta = {
        "num_reconstructions": 1,
        "registered_images": actual_n_views,
        "total_points": n_points,
        "sfm_backend": "mast3r",
        "elapsed_sec": round(elapsed, 1),
    }
    return sparse_dir, meta


# ---------- Image loading helpers ----------

def _get_sorted_images(images_dir: Path) -> list[str]:
    """Get sorted image file paths from a directory."""
    import re
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [
        str(f) for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]

    def extract_num(path):
        match = re.search(r'\d+', Path(path).stem)
        return int(match.group()) if match else float('inf')

    return sorted(files, key=extract_num)


def _load_images_for_mast3r(
    image_files: list[str], size: int
) -> tuple[list[dict], tuple[int, int]]:
    """Load images in DUSt3R/MASt3R expected format."""
    import PIL.Image
    from PIL.ImageOps import exif_transpose
    import torchvision.transforms as tvf

    ImgNorm = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    _ensure_mast3r_on_path()
    from dust3r.utils.image import _resize_pil_image

    imgs = []
    org_w, org_h = None, None

    for path in image_files:
        img = exif_transpose(PIL.Image.open(path)).convert('RGB')
        W1, H1 = img.size
        if org_w is None:
            org_w, org_h = W1, H1

        # Resize long side to `size`
        img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        imgs.append(dict(
            img=ImgNorm(img)[None],
            true_shape=np.int32([img.size[::-1]]),
            idx=len(imgs),
            instance=str(len(imgs)),
        ))

    return imgs, (org_w, org_h)


# ---------- Co-visibility computation ----------

def _compute_co_vis_masks(
    sorted_indices, depthmaps, pointmaps, intrinsics,
    extrinsics_w2c, image_sizes, depth_threshold
):
    """Compute co-visibility masks between views."""
    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    masks = np.zeros((num_images, h, w), dtype=bool)

    for i, idx in enumerate(sorted_indices):
        if i == 0:
            continue

        prev_indices = sorted_indices[:i]
        pts_prev = pointmaps[prev_indices].reshape(-1, 3)
        depths_prev = depthmaps[prev_indices].reshape(-1)
        curr_depth = depthmaps[idx].reshape(h, w)

        # Normalize depths
        d_min, d_max = depths_prev.min(), depths_prev.max()
        if d_max > d_min:
            depths_prev = (depths_prev - d_min) / (d_max - d_min)
        cd_min, cd_max = curr_depth.min(), curr_depth.max()
        if cd_max > cd_min:
            curr_depth = (curr_depth - cd_min) / (cd_max - cd_min)

        # Project previous points into current view
        pts_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1))])
        pts_cam = (extrinsics_w2c[idx] @ pts_h.T).T
        pts_2d_h = (intrinsics[idx] @ pts_cam[:, :3].T).T
        pts_2d = pts_2d_h[:, :2] / (pts_2d_h[:, 2:] + 1e-8)

        valid = (
            (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        )
        vp = pts_2d[valid].astype(int)
        vd = depths_prev[valid]
        x, y = vp[:, 0], vp[:, 1]

        depth_diff = np.abs(vd - curr_depth[y, x])
        consistent = depth_diff < depth_threshold
        masks[idx][y[consistent], x[consistent]] = True

    return masks


# ---------- COLMAP binary format writers ----------

def _write_cameras_bin(path: Path, n_cameras: int, width: int, height: int, fx: float, fy: float):
    """Write cameras.bin in COLMAP binary format (PINHOLE model)."""
    PINHOLE_ID = 1
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n_cameras))
        for cam_id in range(1, n_cameras + 1):
            f.write(struct.pack("<I", cam_id))       # camera_id
            f.write(struct.pack("<i", PINHOLE_ID))    # model_id (PINHOLE=1)
            f.write(struct.pack("<Q", width))         # width
            f.write(struct.pack("<Q", height))        # height
            # PINHOLE params: fx, fy, cx, cy
            f.write(struct.pack("<4d", fx, fy, width / 2.0, height / 2.0))


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _write_images_bin(path: Path, extrinsics_w2c: np.ndarray, image_files: list[str]):
    """Write images.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(extrinsics_w2c)))
        for i, (w2c, img_path) in enumerate(zip(extrinsics_w2c, image_files)):
            img_id = i + 1
            cam_id = i + 1
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = _rotmat_to_qvec(R)
            name = Path(img_path).name

            f.write(struct.pack("<I", img_id))           # image_id
            f.write(struct.pack("<4d", *qvec))           # qvec (w, x, y, z)
            f.write(struct.pack("<3d", *t))              # tvec
            f.write(struct.pack("<I", cam_id))           # camera_id
            # Image name as null-terminated string
            f.write(name.encode("utf-8") + b"\x00")
            # Number of 2D points (0 — we don't store feature matches)
            f.write(struct.pack("<Q", 0))


def _write_points3d_bin(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write points3D.bin in COLMAP binary format."""
    n_points = len(points)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n_points))
        for i in range(n_points):
            point_id = i + 1
            xyz = points[i]
            rgb = colors[i]
            error = 0.0
            f.write(struct.pack("<Q", point_id))         # point3D_id
            f.write(struct.pack("<3d", *xyz))            # xyz
            f.write(struct.pack("<3B", *rgb))            # rgb
            f.write(struct.pack("<d", error))            # error
            f.write(struct.pack("<Q", 0))                # track_length (0)
