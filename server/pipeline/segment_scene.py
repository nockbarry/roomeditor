"""Scene segmentation pipeline using Gaussian Grouping + SAM 2.

This module will be implemented in Phase 2. For now it provides
a stub that creates a uniform segment map (all Gaussians in one segment).
"""

import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


async def segment_scene(
    ply_path: Path,
    images_dir: Path,
    output_seg_path: Path,
) -> list[dict]:
    """Segment a trained Gaussian scene into objects.

    Returns a list of object metadata dicts with:
        segment_id, label, gaussian_start, gaussian_count, centroid, bbox_min, bbox_max

    Phase 2 will integrate Gaussian Grouping + SAM 2.
    For now, returns a single segment containing all Gaussians.
    """
    # Count Gaussians in PLY
    gaussian_count = _count_ply_vertices(ply_path)

    # Write a uniform segment map (all segment 0)
    seg_data = np.zeros(gaussian_count, dtype=np.uint16)
    seg_data.tofile(output_seg_path)

    logger.info(f"Wrote placeholder segment map ({gaussian_count} Gaussians) to {output_seg_path}")

    return [{
        "segment_id": 0,
        "label": "room",
        "gaussian_start": 0,
        "gaussian_count": gaussian_count,
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
