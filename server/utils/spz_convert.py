"""PLY -> SPZ conversion and positions sidecar generation.

SPZ is Niantic's compressed Gaussian Splatting format (~10-24x smaller than PLY).
Spark.js loads SPZ natively via SplatMesh({ fileBytes }) with auto-detection.

The positions sidecar is a compact binary file containing just XYZ positions +
precomputed centroid and radius, enabling fast KD-tree construction on the client
without parsing the full scene file.
"""

import logging
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def ply_to_spz(ply_path: Path, spz_path: Path) -> dict:
    """Convert a 3DGS PLY file to SPZ format.

    Args:
        ply_path: Source PLY file path
        spz_path: Destination SPZ file path

    Returns:
        dict with n_gaussians, ply_size, spz_size, compression_ratio, duration_sec
    """
    import spz as spz_lib

    t0 = time.time()
    ply_size = ply_path.stat().st_size

    cloud = spz_lib.load_splat_from_ply(str(ply_path))
    n_gaussians = cloud.num_points

    # Write SPZ via temp file + rename (same safe-write pattern as PLY)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".spz", dir=str(spz_path.parent))
    try:
        os.close(tmp_fd)
        spz_lib.save_spz(cloud, spz_lib.PackOptions(), tmp_path)
        shutil.move(tmp_path, str(spz_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    spz_size = spz_path.stat().st_size
    duration = time.time() - t0
    ratio = ply_size / spz_size if spz_size > 0 else 0

    logger.info(
        f"PLY->SPZ: {n_gaussians:,} gaussians, "
        f"{ply_size / 1e6:.1f}MB -> {spz_size / 1e6:.1f}MB ({ratio:.1f}x), "
        f"{duration:.1f}s"
    )

    return {
        "n_gaussians": n_gaussians,
        "ply_size": ply_size,
        "spz_size": spz_size,
        "compression_ratio": round(ratio, 1),
        "duration_sec": round(duration, 1),
    }


def extract_positions_sidecar(ply_path: Path, positions_path: Path) -> dict:
    """Extract positions from PLY into a compact binary sidecar file.

    Binary format:
        uint32   N              (gaussian count)
        float32  cx, cy, cz     (centroid)
        float32  radius         (bounding radius)
        float32  x0,y0,z0, x1,y1,z1, ...  (N*3 positions, tightly packed)

    Total: 20 + N*12 bytes. For 4.6M gaussians: ~55MB.
    """
    from plyfile import PlyData

    t0 = time.time()

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n = len(vertex.data)

    positions = np.stack(
        [np.array(vertex["x"]), np.array(vertex["y"]), np.array(vertex["z"])],
        axis=-1,
    ).astype(np.float32)

    # Compute centroid and bounding radius (sampled for speed)
    sample_step = max(1, n // 2000)
    sampled = positions[::sample_step]
    centroid = sampled.mean(axis=0).astype(np.float32)
    diffs = sampled - centroid
    radius = float(np.sqrt((diffs**2).sum(axis=1).max()))

    # Write binary file via temp + rename
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".positions.bin", dir=str(positions_path.parent))
    try:
        os.close(tmp_fd)
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<I", n))
            f.write(struct.pack("<3f", centroid[0], centroid[1], centroid[2]))
            f.write(struct.pack("<f", radius))
            f.write(positions.tobytes())
        shutil.move(tmp_path, str(positions_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # Free memory
    del plydata, vertex, positions

    file_size = positions_path.stat().st_size
    duration = time.time() - t0

    logger.info(
        f"Positions sidecar: {n:,} gaussians, {file_size / 1e6:.1f}MB, {duration:.1f}s"
    )

    return {
        "n_gaussians": n,
        "centroid": centroid.tolist(),
        "radius": radius,
        "file_size": file_size,
        "duration_sec": round(duration, 1),
    }


def generate_spz_bundle(project_dir: Path) -> dict | None:
    """Generate both scene.spz and scene.positions.bin from scene.ply.

    This is the single entry point called after any PLY mutation.
    Returns None if scene.ply doesn't exist.
    Errors are logged but do NOT propagate — SPZ is best-effort.
    """
    ply_path = project_dir / "scene.ply"
    if not ply_path.exists():
        return None

    spz_path = project_dir / "scene.spz"
    positions_path = project_dir / "scene.positions.bin"

    result = {}

    try:
        result["spz"] = ply_to_spz(ply_path, spz_path)
    except Exception as e:
        logger.error(f"SPZ conversion failed: {e}", exc_info=True)
        if spz_path.exists():
            spz_path.unlink()

    try:
        result["positions"] = extract_positions_sidecar(ply_path, positions_path)
    except Exception as e:
        logger.error(f"Positions sidecar generation failed: {e}", exc_info=True)
        if positions_path.exists():
            positions_path.unlink()

    return result


def invalidate_spz(project_dir: Path):
    """Delete stale SPZ + positions files."""
    for name in ("scene.spz", "scene.positions.bin"):
        p = project_dir / name
        if p.exists():
            p.unlink()
