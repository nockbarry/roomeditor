"""Gaussian splat compression utilities.

Handles PLY optimization (pruning low-opacity Gaussians) and
conversion to compact formats.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


async def prune_gaussians(
    input_ply: Path,
    output_ply: Path,
    opacity_threshold: float = 0.005,
    max_gaussians: int = 2_000_000,
) -> dict:
    """Remove low-opacity Gaussians and cap total count.

    Returns dict with before/after counts and stats.
    """
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(str(input_ply))
    vertex = plydata["vertex"]
    n_before = len(vertex.data)

    if n_before == 0:
        return {"n_before": 0, "n_after": 0, "n_pruned": 0}

    # Extract opacities — check if stored as logits or raw
    if "opacity" in vertex.data.dtype.names:
        opa_raw = np.array(vertex["opacity"])
        # Standard 3DGS stores as logits; AnySplat may store raw [0,1]
        # If values are mostly in [-10, 10], they're logits
        if np.abs(opa_raw).max() > 1.5:
            opacities = 1.0 / (1.0 + np.exp(-opa_raw))
        else:
            opacities = opa_raw
    else:
        # No opacity field — keep all
        if input_ply != output_ply:
            import shutil
            shutil.copy2(input_ply, output_ply)
        return {"n_before": n_before, "n_after": n_before, "n_pruned": 0}

    # Filter by opacity threshold
    keep_mask = opacities > opacity_threshold
    n_keep = int(keep_mask.sum())

    # If over max, keep highest-opacity ones
    if n_keep > max_gaussians:
        indices = np.where(keep_mask)[0]
        top_indices = indices[np.argsort(opacities[indices])[-max_gaussians:]]
        keep_mask = np.zeros(n_before, dtype=bool)
        keep_mask[top_indices] = True
        n_keep = max_gaussians

    # Write filtered PLY
    filtered_data = vertex.data[keep_mask]
    el = PlyElement.describe(filtered_data, "vertex")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(output_ply))

    n_pruned = n_before - n_keep
    logger.info(
        f"Pruned {n_pruned:,} Gaussians ({n_before:,} → {n_keep:,}), "
        f"opacity threshold={opacity_threshold}"
    )
    return {
        "n_before": n_before,
        "n_after": n_keep,
        "n_pruned": n_pruned,
    }
