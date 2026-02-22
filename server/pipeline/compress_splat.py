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


async def deduplicate_gaussians(
    ply_path: Path,
    distance_threshold: float = 0.005,
) -> dict:
    """Remove duplicate Gaussians that appear in chunk overlap regions.

    Uses a KD-tree to find pairs of Gaussians closer than distance_threshold.
    For each pair, keeps the one with higher opacity and removes the other.

    Args:
        ply_path: Path to PLY file (modified in place)
        distance_threshold: Max distance to consider as duplicate (meters)

    Returns:
        dict with n_before, n_after, n_removed
    """
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n_before = len(vertex.data)

    if n_before < 2:
        return {"n_before": n_before, "n_after": n_before, "n_removed": 0}

    # Extract positions
    positions = np.column_stack([
        np.array(vertex["x"]),
        np.array(vertex["y"]),
        np.array(vertex["z"]),
    ])

    # Extract opacities for comparison
    if "opacity" in vertex.data.dtype.names:
        opacities = np.array(vertex["opacity"])
    else:
        opacities = np.zeros(n_before)

    # Build KD-tree and find close pairs
    tree = cKDTree(positions)
    pairs = tree.query_pairs(distance_threshold)

    if not pairs:
        logger.info(f"Dedup: no duplicates found within {distance_threshold}m")
        return {"n_before": n_before, "n_after": n_before, "n_removed": 0}

    # For each pair, mark the lower-opacity one for removal
    to_remove = set()
    for i, j in pairs:
        if i in to_remove or j in to_remove:
            continue
        # Keep higher opacity, remove lower
        if opacities[i] >= opacities[j]:
            to_remove.add(j)
        else:
            to_remove.add(i)

    # Build keep mask
    keep_mask = np.ones(n_before, dtype=bool)
    for idx in to_remove:
        keep_mask[idx] = False

    n_removed = len(to_remove)
    n_after = n_before - n_removed

    if n_removed > 0:
        filtered_data = vertex.data[keep_mask]
        el = PlyElement.describe(filtered_data, "vertex")
        PlyData([el]).write(str(ply_path))

    logger.info(
        f"Dedup: removed {n_removed:,} duplicates ({n_before:,} -> {n_after:,}), "
        f"threshold={distance_threshold}m"
    )
    return {
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_removed,
    }
