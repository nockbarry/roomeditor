"""Spherical harmonics coefficient manipulation for per-segment lighting."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def modify_segment_lighting(
    ply_path: Path,
    gaussian_ids: list[int],
    brightness: float = 1.0,
    color_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sh_scale: float = 1.0,
):
    """Modify SH coefficients in the PLY file for specified gaussians.

    Args:
        ply_path: Path to the PLY file
        gaussian_ids: List of gaussian indices to modify
        brightness: Multiply SH DC component (overall brightness)
        color_tint: RGB multiplier on DC (color shift)
        sh_scale: Scale higher SH bands (view-dependence/specularity)
    """
    from plyfile import PlyData
    import tempfile, shutil, os

    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]
    ids = np.array(gaussian_ids)

    # Modify DC SH coefficients (brightness and color)
    dc_attrs = ["f_dc_0", "f_dc_1", "f_dc_2"]
    for ch, attr in enumerate(dc_attrs):
        if attr in vertices.data.dtype.names:
            vals = np.array(vertices[attr], dtype=np.float64)
            vals[ids] *= brightness * color_tint[ch]
            vertices[attr] = vals.astype(np.float32)

    # Scale higher-order SH bands (rest coefficients)
    if sh_scale != 1.0:
        for i in range(45):  # Up to SH degree 3
            attr = f"f_rest_{i}"
            if attr in vertices.data.dtype.names:
                vals = np.array(vertices[attr], dtype=np.float64)
                vals[ids] *= sh_scale
                vertices[attr] = vals.astype(np.float32)

    # Safe write via temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=str(ply_path.parent))
    try:
        os.close(tmp_fd)
        plydata.write(tmp_path)
        shutil.move(tmp_path, str(ply_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(
        f"Modified lighting for {len(ids)} gaussians: "
        f"brightness={brightness}, tint={color_tint}, sh_scale={sh_scale}"
    )
