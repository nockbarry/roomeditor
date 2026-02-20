"""Three-pass automatic gaussian cleanup pipeline.

Pass 1 — Opacity: Remove gaussians with sigmoid(opacity) < 0.005
Pass 2 — Scale outliers: Remove gaussians where any log-scale axis is >3σ above mean
Pass 3 — Spatial floaters: KDTree, remove gaussians whose NN distance > 5× median NN distance
                           (except high-opacity >0.8 gaussians)
"""

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData

logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def auto_cleanup(ply_path: Path) -> dict:
    """Run three-pass cleanup on a PLY file in-place.

    Returns stats dict with n_before, n_after, and per-pass removal counts.
    """
    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]
    n_before = len(vertices.data)

    # Build a boolean mask: True = keep
    keep = np.ones(n_before, dtype=bool)

    # --- Pass 1: Opacity ---
    n_opacity_removed = 0
    if "opacity" in vertices.data.dtype.names:
        raw_opacity = np.array(vertices["opacity"], dtype=np.float64)
        opa = sigmoid(raw_opacity)
        opacity_mask = opa >= 0.005
        n_opacity_removed = int(np.sum(~opacity_mask & keep))
        keep &= opacity_mask
        logger.info(f"Cleanup pass 1 (opacity): removed {n_opacity_removed}")

    # --- Pass 2: Scale outliers ---
    n_scale_removed = 0
    scale_names = [n for n in vertices.data.dtype.names if n.startswith("scale_")]
    if scale_names and np.sum(keep) > 100:
        scale_outlier = np.zeros(n_before, dtype=bool)
        for sn in scale_names:
            vals = np.array(vertices[sn], dtype=np.float64)
            # Compute stats only on currently-kept gaussians
            kept_vals = vals[keep]
            mean = kept_vals.mean()
            std = kept_vals.std()
            if std > 0:
                scale_outlier |= vals > (mean + 3.0 * std)
        n_scale_removed = int(np.sum(scale_outlier & keep))
        keep &= ~scale_outlier
        logger.info(f"Cleanup pass 2 (scale): removed {n_scale_removed}")

    # --- Pass 3: Spatial floaters ---
    n_floater_removed = 0
    if np.sum(keep) > 100 and all(n in vertices.data.dtype.names for n in ("x", "y", "z")):
        from scipy.spatial import cKDTree

        x = np.array(vertices["x"], dtype=np.float64)
        y = np.array(vertices["y"], dtype=np.float64)
        z = np.array(vertices["z"], dtype=np.float64)
        positions = np.column_stack([x, y, z])

        # Build KDTree on kept positions only
        kept_indices = np.where(keep)[0]
        kept_positions = positions[kept_indices]
        tree = cKDTree(kept_positions)

        # Query nearest neighbor (k=2 because first is self)
        dists, _ = tree.query(kept_positions, k=2)
        nn_dists = dists[:, 1]  # distance to nearest neighbor

        median_nn = np.median(nn_dists)
        threshold = 5.0 * median_nn

        # Determine high-opacity gaussians (exempt from floater removal)
        if "opacity" in vertices.data.dtype.names:
            raw_opacity = np.array(vertices["opacity"], dtype=np.float64)
            high_opa = sigmoid(raw_opacity[kept_indices]) > 0.8
        else:
            high_opa = np.zeros(len(kept_indices), dtype=bool)

        is_floater = (nn_dists > threshold) & ~high_opa
        floater_global_ids = kept_indices[is_floater]
        n_floater_removed = len(floater_global_ids)
        keep[floater_global_ids] = False
        logger.info(f"Cleanup pass 3 (floaters): removed {n_floater_removed}")

    # --- Write filtered PLY ---
    n_after = int(np.sum(keep))
    if n_after < n_before:
        filtered_data = vertices.data[keep]
        from plyfile import PlyElement

        new_element = PlyElement.describe(filtered_data, "vertex")
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=str(ply_path.parent))
        import os
        os.close(tmp_fd)
        try:
            PlyData([new_element], text=False).write(tmp_path)
            shutil.move(tmp_path, str(ply_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        logger.info(f"Cleanup complete: {n_before} → {n_after} gaussians")
    else:
        logger.info("Cleanup: no gaussians removed")

    return {
        "n_before": n_before,
        "n_after": n_after,
        "n_opacity_removed": n_opacity_removed,
        "n_scale_removed": n_scale_removed,
        "n_floater_removed": n_floater_removed,
    }
