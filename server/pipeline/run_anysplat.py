"""AnySplat feed-forward 3D Gaussian Splatting from unposed images.

Unlike COLMAP or MASt3R pipelines that estimate poses then train Gaussians,
AnySplat produces both camera poses AND Gaussians in a single forward pass.
This makes it dramatically faster (seconds vs minutes) and eliminates the
multi-cluster SfM failure mode entirely.

Trained on ARKitScenes + ScanNet++ indoor datasets — ideal for room scanning.

Requirements:
    - AnySplat installed at /home/nock/projects/anysplat/
    - Pre-trained weights (auto-downloaded on first run)
    - ~16GB VRAM for up to 16 views at 448x448

Usage:
    from pipeline.run_anysplat import run_anysplat
    ply_path, meta = run_anysplat(images_dir, output_dir)
"""

import logging
import sys
from pathlib import Path
from time import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

ANYSPLAT_DIR = Path("/home/nock/projects/anysplat")


def _ensure_anysplat_on_path():
    """Add AnySplat to sys.path."""
    p = str(ANYSPLAT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def run_anysplat(
    images_dir: Path,
    output_dir: Path,
    max_views: int = 16,
    image_size: int = 448,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict]:
    """Run AnySplat feed-forward reconstruction.

    This replaces the entire COLMAP/MASt3R + Gaussian training pipeline
    with a single forward pass through AnySplat.

    Args:
        images_dir: Directory containing input images.
        output_dir: Output directory for results.
        max_views: Maximum number of views to process (VRAM limit).
        image_size: Input image size (AnySplat uses 448).
        progress_callback: Optional progress callback (0-1).

    Returns:
        (ply_path, metadata) where ply_path is the output PLY file.
    """
    import torch

    _ensure_anysplat_on_path()

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "scene.ply"

    if progress_callback:
        progress_callback(0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time()

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    n_total = len(image_files)

    # Subsample for VRAM
    if n_total > max_views:
        step = max(1, n_total // max_views)
        image_files = image_files[::step][:max_views]
        logger.info(f"AnySplat: subsampled {n_total} -> {len(image_files)} views")

    n_views = len(image_files)
    logger.info(f"AnySplat: processing {n_views} images")

    if progress_callback:
        progress_callback(0.1)

    # Load model
    logger.info("AnySplat: loading model...")
    from src.model.model.anysplat import AnySplat

    model = AnySplat.from_pretrained("anysplat_ckpt_v1").to(device).eval()

    if progress_callback:
        progress_callback(0.3)

    # Preprocess images
    logger.info("AnySplat: preprocessing images...")
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    images = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        images.append(transform(img))

    batch = torch.stack(images).unsqueeze(0).to(device)

    if progress_callback:
        progress_callback(0.4)

    # Run inference
    logger.info("AnySplat: running inference...")
    with torch.no_grad():
        output = model(batch)

    if progress_callback:
        progress_callback(0.8)

    # Export PLY
    logger.info("AnySplat: exporting PLY...")
    try:
        if hasattr(output, "export_ply"):
            output.export_ply(str(ply_path))
        elif hasattr(output, "save"):
            output.save(str(ply_path))
        else:
            # Manual export from Gaussian parameters
            _export_gaussians_ply(output, ply_path)
    except Exception as e:
        logger.error(f"AnySplat PLY export failed: {e}")
        raise

    elapsed = time() - t_start

    # Get stats
    n_gaussians = 0
    if hasattr(output, "gaussians"):
        gs = output.gaussians
        if hasattr(gs, "shape"):
            n_gaussians = gs.shape[-2] if len(gs.shape) > 1 else gs.shape[0]
    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # Cleanup
    del model, output, batch
    torch.cuda.empty_cache()

    if progress_callback:
        progress_callback(1.0)

    meta = {
        "sfm_backend": "anysplat",
        "n_views": n_views,
        "n_gaussians": n_gaussians,
        "elapsed_sec": round(elapsed, 1),
        "vram_peak_gb": round(vram_peak, 1),
    }
    logger.info(
        f"AnySplat complete: {n_views} views -> {n_gaussians} Gaussians "
        f"in {elapsed:.1f}s (VRAM peak: {vram_peak:.1f}GB)"
    )

    return ply_path, meta


def _export_gaussians_ply(output, ply_path: Path):
    """Export Gaussian parameters to PLY format.

    Handles the common output structure from feed-forward models.
    """
    import struct

    # Try to extract parameters from different output formats
    means = None
    colors = None
    opacities = None
    scales = None
    rotations = None

    if hasattr(output, "means"):
        means = output.means.detach().cpu().numpy().reshape(-1, 3)
    elif hasattr(output, "xyz"):
        means = output.xyz.detach().cpu().numpy().reshape(-1, 3)

    if hasattr(output, "colors"):
        colors = output.colors.detach().cpu().numpy().reshape(-1, 3)
    elif hasattr(output, "rgb"):
        colors = output.rgb.detach().cpu().numpy().reshape(-1, 3)

    if hasattr(output, "opacities"):
        opacities = output.opacities.detach().cpu().numpy().reshape(-1)
    elif hasattr(output, "opacity"):
        opacities = output.opacity.detach().cpu().numpy().reshape(-1)

    if hasattr(output, "scales"):
        scales = output.scales.detach().cpu().numpy().reshape(-1, 3)
    elif hasattr(output, "scale"):
        scales = output.scale.detach().cpu().numpy().reshape(-1, 3)

    if hasattr(output, "rotations"):
        rotations = output.rotations.detach().cpu().numpy().reshape(-1, 4)
    elif hasattr(output, "rotation"):
        rotations = output.rotation.detach().cpu().numpy().reshape(-1, 4)

    if means is None:
        raise ValueError("Cannot extract Gaussian means from AnySplat output")

    n = len(means)
    if colors is None:
        colors = np.ones((n, 3)) * 0.5
    if opacities is None:
        opacities = np.ones(n) * 0.8
    if scales is None:
        scales = np.ones((n, 3)) * 0.01
    if rotations is None:
        rotations = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)

    # Clamp colors to [0, 1]
    colors = np.clip(colors, 0, 1)

    # Write standard 3DGS PLY
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    # Convert SH DC from RGB
    sh_c0 = 0.28209479177387814
    f_dc = (colors - 0.5) / sh_c0

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            # xyz
            f.write(struct.pack("<3f", *means[i]))
            # normals (zeros)
            f.write(struct.pack("<3f", 0, 0, 0))
            # f_dc (SH coefficients)
            f.write(struct.pack("<3f", *f_dc[i]))
            # opacity (logit)
            opa_logit = np.log(opacities[i] / (1 - opacities[i] + 1e-8))
            f.write(struct.pack("<f", opa_logit))
            # scales (log)
            log_scales = np.log(np.abs(scales[i]) + 1e-8)
            f.write(struct.pack("<3f", *log_scales))
            # rotations (quaternion wxyz)
            f.write(struct.pack("<4f", *rotations[i]))

    logger.info(f"Exported {n} Gaussians to {ply_path}")
