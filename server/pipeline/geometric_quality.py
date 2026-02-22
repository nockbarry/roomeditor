"""Geometric quality measures for 3D Gaussian Splatting reconstructions.

Evaluates reconstruction geometry using rendered depth maps and alpha coverage,
without requiring ground-truth geometry. These measures complement pixel-level
PSNR/SSIM which require perfect camera alignment to be meaningful.

Measures:
    1. Multi-view depth consistency — do depth maps agree across views?
    2. Depth-RGB edge alignment — do depth edges match color edges?
    3. Alpha coverage — are there holes or poorly-covered regions?
    4. Composite health score — weighted combination (0-100 scale)
"""

import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PLY / Camera loading helpers
# ---------------------------------------------------------------------------

def load_ply_as_params(ply_path: Path) -> tuple[dict, int]:
    """Load a 3DGS PLY file into rasterizer-compatible params dict.

    Returns:
        (params, sh_degree) where params has keys:
        means, quats, scales, opacities, sh0, shN
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    names = vertex.data.dtype.names

    device = torch.device("cuda")

    means = torch.tensor(
        np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32),
        device=device,
    )
    quats = torch.tensor(
        np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=-1,
        ).astype(np.float32),
        device=device,
    )
    scales = torch.tensor(
        np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
        ).astype(np.float32),
        device=device,
    )
    opacities = torch.tensor(
        np.array(vertex["opacity"]).astype(np.float32),
        device=device,
    ).unsqueeze(-1)

    sh0 = torch.tensor(
        np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
        ).astype(np.float32),
        device=device,
    ).unsqueeze(1)  # (N, 1, 3)

    # Count f_rest fields to determine SH degree
    rest_count = sum(1 for n in names if n.startswith("f_rest_"))
    if rest_count > 0:
        rest_data = np.stack(
            [np.array(vertex[f"f_rest_{i}"]).astype(np.float32) for i in range(rest_count)],
            axis=-1,
        )
        # rest_count = 3 * ((L+1)^2 - 1), solve for L
        n_coeffs_per_channel = rest_count // 3
        sh_degree = int(np.sqrt(n_coeffs_per_channel + 1)) - 1
        shN = torch.tensor(
            rest_data.reshape(-1, 3, n_coeffs_per_channel),
            device=device,
        )  # (N, 3, K-1)
        # Reshape to (N, K-1, 3) to match gsplat's expected layout
        shN = shN.permute(0, 2, 1)  # (N, K-1, 3)
    else:
        sh_degree = 0
        n_pts = means.shape[0]
        shN = torch.zeros(n_pts, 0, 3, device=device)

    params = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }

    return params, sh_degree


def load_cameras_from_json(cameras_path: Path) -> list[dict]:
    """Load cameras.json into a list of camera dicts with viewmat/K arrays.

    Returns list of dicts with keys: viewmat, K, width, height, frame
    """
    with open(cameras_path) as f:
        cam_data = json.load(f)

    resolution = cam_data.get("resolution", 448)
    default_focal = cam_data.get("focal_length_px", resolution * 0.85)
    cameras = []

    for cam in cam_data["cameras"]:
        transform = np.array(cam["transform"], dtype=np.float32)
        if transform.shape == (3, 4):
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :] = transform
        else:
            c2w = transform

        viewmat = np.linalg.inv(c2w).astype(np.float32)

        fx = cam.get("fx", default_focal)
        fy = cam.get("fy", default_focal)
        cx = cam.get("cx", resolution / 2)
        cy = cam.get("cy", resolution / 2)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        cameras.append({
            "viewmat": viewmat,
            "K": K,
            "width": resolution,
            "height": resolution,
            "frame": cam.get("frame", ""),
        })

    return cameras


def _select_eval_views(cameras: list[dict], n_views: int) -> list[int]:
    """Select evenly-spaced view indices for evaluation."""
    n_total = len(cameras)
    if n_total <= n_views:
        return list(range(n_total))
    step = n_total / n_views
    return [int(i * step) for i in range(n_views)]


def _select_nearby_pairs(cameras: list[dict], n_pairs: int) -> list[tuple[int, int]]:
    """Select nearby camera pairs using KD-tree on camera centers."""
    positions = []
    for cam in cameras:
        viewmat = cam["viewmat"]
        c2w = np.linalg.inv(viewmat)
        positions.append(c2w[:3, 3])
    positions = np.array(positions)

    tree = cKDTree(positions)
    # For each camera, get nearest neighbor
    dists, nn = tree.query(positions, k=2)

    # Build unique pairs sorted by distance
    pairs = set()
    for i in range(len(positions)):
        j = nn[i, 1]  # nearest neighbor (k=0 is self)
        pair = (min(i, j), max(i, j))
        pairs.add(pair)

    pairs = sorted(pairs)
    if len(pairs) > n_pairs:
        step = len(pairs) / n_pairs
        pairs = [pairs[int(i * step)] for i in range(n_pairs)]

    return pairs


def _render_view(params, rasterizer, camera, sh_degree):
    """Render a single view, returns (rgb, depth, alpha) as tensors."""
    device = params["means"].device
    viewmat = torch.tensor(camera["viewmat"], device=device).unsqueeze(0)
    K = torch.tensor(camera["K"], device=device).unsqueeze(0)
    W, H = camera["width"], camera["height"]

    result = rasterizer.rasterize(params, viewmat, K, W, H, sh_degree)
    return result.image, result.depth, result.alphas


# ---------------------------------------------------------------------------
# Measure 1: Multi-View Depth Consistency
# ---------------------------------------------------------------------------

def compute_depth_consistency(
    params: dict,
    cameras: list[dict],
    rasterizer,
    sh_degree: int,
    n_pairs: int = 20,
    depth_threshold: float = 0.05,
) -> dict:
    """Check if depth maps agree across nearby views.

    For pairs of nearby views:
    1. Render depth + alpha from view A and view B
    2. Unproject valid pixels from A into 3D
    3. Project those 3D points into view B
    4. Compare rendered depth at projected locations
    5. Report fraction of consistent depths

    Returns dict with pct_consistent, mean_abs_depth_error, median_abs_depth_error
    """
    device = params["means"].device
    pairs = _select_nearby_pairs(cameras, n_pairs)

    all_consistent = []
    all_abs_errors = []

    for idx_a, idx_b in pairs:
        cam_a, cam_b = cameras[idx_a], cameras[idx_b]

        with torch.no_grad():
            _, depth_a, alpha_a = _render_view(params, rasterizer, cam_a, sh_degree)
            _, depth_b, alpha_b = _render_view(params, rasterizer, cam_b, sh_degree)

        H, W = depth_a.shape[:2]

        # Valid pixels in view A (good alpha coverage)
        valid_a = (alpha_a.squeeze(-1) > 0.5)

        # Subsample every 4th pixel for speed
        yy, xx = torch.meshgrid(
            torch.arange(0, H, 4, device=device),
            torch.arange(0, W, 4, device=device),
            indexing="ij",
        )
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        # Filter to valid pixels
        mask = valid_a[yy, xx]
        if mask.sum() < 10:
            continue
        yy = yy[mask]
        xx = xx[mask]

        # Get depths at selected pixels
        d_a = depth_a[yy, xx, 0]

        # Unproject to 3D using K_A^-1 and c2w_A
        K_a = torch.tensor(cam_a["K"], device=device, dtype=torch.float32)
        K_a_inv = torch.linalg.inv(K_a)
        c2w_a = torch.linalg.inv(
            torch.tensor(cam_a["viewmat"], device=device, dtype=torch.float32)
        )

        # Pixel coords -> camera coords
        px = torch.stack([xx.float(), yy.float(), torch.ones_like(xx, dtype=torch.float32)], dim=-1)
        cam_pts = (K_a_inv @ px.unsqueeze(-1)).squeeze(-1) * d_a.unsqueeze(-1)

        # Camera coords -> world coords
        world_pts = (c2w_a[:3, :3] @ cam_pts.unsqueeze(-1)).squeeze(-1) + c2w_a[:3, 3]

        # Project into view B
        w2c_b = torch.tensor(cam_b["viewmat"], device=device, dtype=torch.float32)
        K_b = torch.tensor(cam_b["K"], device=device, dtype=torch.float32)

        cam_pts_b = (w2c_b[:3, :3] @ world_pts.unsqueeze(-1)).squeeze(-1) + w2c_b[:3, 3]
        z_b = cam_pts_b[:, 2]

        # Filter points behind camera B
        in_front = z_b > 0.01
        if in_front.sum() < 10:
            continue

        cam_pts_b = cam_pts_b[in_front]
        z_b = z_b[in_front]

        # Project to pixel coords in B
        px_b = (K_b @ cam_pts_b.unsqueeze(-1)).squeeze(-1)
        u_b = px_b[:, 0] / z_b
        v_b = px_b[:, 1] / z_b

        # Filter to within image bounds
        in_bounds = (u_b >= 0) & (u_b < W - 1) & (v_b >= 0) & (v_b < H - 1)
        if in_bounds.sum() < 10:
            continue

        u_b = u_b[in_bounds]
        v_b = v_b[in_bounds]
        z_b = z_b[in_bounds]

        # Sample rendered depth from B using grid_sample
        depth_b_hw = depth_b.squeeze(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grid_x = 2.0 * u_b / (W - 1) - 1.0
        grid_y = 2.0 * v_b / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        sampled_depth = F.grid_sample(
            depth_b_hw, grid, mode="bilinear", align_corners=True
        ).squeeze()

        # Also check alpha at sampled locations
        alpha_b_hw = alpha_b.squeeze(-1).unsqueeze(0).unsqueeze(0)
        sampled_alpha = F.grid_sample(
            alpha_b_hw, grid, mode="bilinear", align_corners=True
        ).squeeze()

        valid = sampled_alpha > 0.5
        if valid.sum() < 10:
            continue

        z_b = z_b[valid]
        sampled_depth = sampled_depth[valid]

        # Relative depth error
        rel_error = torch.abs(z_b - sampled_depth) / (sampled_depth + 1e-6)
        consistent = (rel_error < depth_threshold).float()

        all_consistent.append(consistent.mean().item())
        all_abs_errors.extend(torch.abs(z_b - sampled_depth).cpu().tolist())

    if not all_consistent:
        return {
            "pct_consistent": 0.0,
            "mean_abs_depth_error": 0.0,
            "median_abs_depth_error": 0.0,
            "n_pairs_evaluated": 0,
        }

    abs_errors = np.array(all_abs_errors)
    return {
        "pct_consistent": float(np.mean(all_consistent)),
        "mean_abs_depth_error": float(abs_errors.mean()),
        "median_abs_depth_error": float(np.median(abs_errors)),
        "n_pairs_evaluated": len(all_consistent),
    }


# ---------------------------------------------------------------------------
# Measure 2: Depth-RGB Edge Alignment
# ---------------------------------------------------------------------------

def compute_edge_alignment(
    params: dict,
    cameras: list[dict],
    rasterizer,
    sh_degree: int,
    n_views: int = 10,
) -> dict:
    """Check if depth edges align with RGB edges.

    For selected views:
    1. Render RGB + depth
    2. Compute gradient magnitude for both
    3. Compute Pearson correlation between depth and RGB gradient magnitudes

    Higher correlation = depth edges align with color edges = good geometry.

    Returns dict with mean_edge_alignment (0-1).
    """
    view_indices = _select_eval_views(cameras, n_views)
    correlations = []

    for idx in view_indices:
        cam = cameras[idx]

        with torch.no_grad():
            rgb, depth, alpha = _render_view(params, rasterizer, cam, sh_degree)

        # Convert RGB to grayscale
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        depth_2d = depth.squeeze(-1)

        # Compute gradient magnitudes using finite differences
        # Pad to handle borders
        gray_pad = F.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        depth_pad = F.pad(depth_2d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")

        # Sobel-like gradients
        gx_gray = gray_pad[0, 0, 1:-1, 2:] - gray_pad[0, 0, 1:-1, :-2]
        gy_gray = gray_pad[0, 0, 2:, 1:-1] - gray_pad[0, 0, :-2, 1:-1]
        grad_gray = torch.sqrt(gx_gray ** 2 + gy_gray ** 2)

        gx_depth = depth_pad[0, 0, 1:-1, 2:] - depth_pad[0, 0, 1:-1, :-2]
        gy_depth = depth_pad[0, 0, 2:, 1:-1] - depth_pad[0, 0, :-2, 1:-1]
        grad_depth = torch.sqrt(gx_depth ** 2 + gy_depth ** 2)

        # Mask to valid regions (good alpha)
        valid = alpha.squeeze(-1) > 0.5
        if valid.sum() < 100:
            continue

        g_rgb = grad_gray[valid].cpu().numpy()
        g_dep = grad_depth[valid].cpu().numpy()

        # Pearson correlation
        if g_rgb.std() < 1e-8 or g_dep.std() < 1e-8:
            continue

        corr = np.corrcoef(g_rgb, g_dep)[0, 1]
        if not np.isnan(corr):
            correlations.append(max(0.0, corr))

    return {
        "mean_edge_alignment": float(np.mean(correlations)) if correlations else 0.0,
        "n_views_evaluated": len(correlations),
    }


# ---------------------------------------------------------------------------
# Measure 3: Alpha Coverage
# ---------------------------------------------------------------------------

def compute_alpha_coverage(
    params: dict,
    cameras: list[dict],
    rasterizer,
    sh_degree: int,
    n_views: int = 10,
) -> dict:
    """Evaluate alpha coverage across views.

    Returns dict with mean_alpha, frac_holes, frac_well_covered, alpha_std_across_views.
    """
    view_indices = _select_eval_views(cameras, n_views)
    mean_alphas = []
    hole_fracs = []
    covered_fracs = []

    for idx in view_indices:
        cam = cameras[idx]

        with torch.no_grad():
            _, _, alpha = _render_view(params, rasterizer, cam, sh_degree)

        a = alpha.squeeze(-1)
        mean_alphas.append(a.mean().item())
        hole_fracs.append((a < 0.5).float().mean().item())
        covered_fracs.append((a > 0.9).float().mean().item())

    if not mean_alphas:
        return {
            "mean_alpha": 0.0,
            "frac_holes": 1.0,
            "frac_well_covered": 0.0,
            "alpha_std_across_views": 0.0,
            "n_views_evaluated": 0,
        }

    return {
        "mean_alpha": float(np.mean(mean_alphas)),
        "frac_holes": float(np.mean(hole_fracs)),
        "frac_well_covered": float(np.mean(covered_fracs)),
        "alpha_std_across_views": float(np.std(mean_alphas)),
        "n_views_evaluated": len(mean_alphas),
    }


# ---------------------------------------------------------------------------
# Measure 4: Composite Health Score
# ---------------------------------------------------------------------------

def compute_health_score(
    gaussian_stats: dict,
    depth_consistency: dict,
    edge_alignment: dict,
    alpha_coverage: dict,
) -> dict:
    """Compute composite health score on a 0-100 scale.

    Weights:
        - Geometry (depth consistency pct_consistent): 30
        - Edge alignment (mean_edge_alignment): 20
        - Coverage (mean_alpha): 20
        - Cleanliness (1 - frac_scale_outlier): 15
        - Opacity health (1 - frac_transparent): 15
    """
    geometry = depth_consistency.get("pct_consistent", 0.0)
    edges = edge_alignment.get("mean_edge_alignment", 0.0)
    coverage = alpha_coverage.get("mean_alpha", 0.0)
    cleanliness = 1.0 - gaussian_stats.get("frac_scale_outlier", 0.0)
    opacity_health = 1.0 - gaussian_stats.get("frac_transparent", 0.0)

    score = (
        30 * geometry
        + 20 * edges
        + 20 * coverage
        + 15 * cleanliness
        + 15 * opacity_health
    )

    return {
        "score": round(float(score), 1),
        "components": {
            "geometry": round(float(geometry), 4),
            "edge_alignment": round(float(edges), 4),
            "coverage": round(float(coverage), 4),
            "cleanliness": round(float(cleanliness), 4),
            "opacity_health": round(float(opacity_health), 4),
        },
        "weights": {
            "geometry": 30,
            "edge_alignment": 20,
            "coverage": 20,
            "cleanliness": 15,
            "opacity_health": 15,
        },
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_geometric_quality(
    ply_path: Path,
    cameras_path: Path,
    n_eval_views: int = 10,
    n_depth_pairs: int = 20,
) -> dict:
    """Run all geometric quality measures and compute composite health score.

    Args:
        ply_path: Path to scene.ply (standard 3DGS format).
        cameras_path: Path to cameras.json.
        n_eval_views: Number of views for edge alignment and alpha coverage.
        n_depth_pairs: Number of view pairs for depth consistency.

    Returns:
        Complete results dict with health_score, depth_consistency,
        edge_alignment, alpha_coverage, health_components, gaussian_stats.
    """
    from pipeline.rasterizer import Rasterizer3DGS
    from pipeline.evaluate_noreference import compute_gaussian_stats

    logger.info(f"Computing geometric quality for {ply_path}")

    # Load PLY
    params, sh_degree = load_ply_as_params(ply_path)
    n_gaussians = params["means"].shape[0]
    logger.info(f"  Loaded {n_gaussians:,} Gaussians (SH degree {sh_degree})")

    # Load cameras
    cameras = load_cameras_from_json(cameras_path)
    logger.info(f"  Loaded {len(cameras)} cameras")

    # Create rasterizer
    rasterizer = Rasterizer3DGS()

    # Gaussian stats (from existing no-reference module)
    gs_stats = compute_gaussian_stats(ply_path)
    gs_dict = {
        "n_gaussians": gs_stats.n_gaussians,
        "n_effective": gs_stats.n_effective,
        "frac_transparent": gs_stats.frac_transparent,
        "frac_opaque": gs_stats.frac_opaque,
        "opacity_mean": gs_stats.opacity_mean,
        "log_scale_mean": gs_stats.log_scale_mean,
        "log_scale_std": gs_stats.log_scale_std,
        "frac_scale_outlier": gs_stats.frac_scale_outlier,
        "mean_nn_dist": gs_stats.mean_nn_dist,
        "bbox_volume": gs_stats.bbox_volume,
        "density": gs_stats.density,
    }

    # Run measures
    logger.info("  Computing depth consistency...")
    depth_result = compute_depth_consistency(
        params, cameras, rasterizer, sh_degree,
        n_pairs=n_depth_pairs,
    )
    logger.info(f"    pct_consistent={depth_result['pct_consistent']:.3f}")

    logger.info("  Computing edge alignment...")
    edge_result = compute_edge_alignment(
        params, cameras, rasterizer, sh_degree,
        n_views=n_eval_views,
    )
    logger.info(f"    mean_edge_alignment={edge_result['mean_edge_alignment']:.3f}")

    logger.info("  Computing alpha coverage...")
    alpha_result = compute_alpha_coverage(
        params, cameras, rasterizer, sh_degree,
        n_views=n_eval_views,
    )
    logger.info(f"    mean_alpha={alpha_result['mean_alpha']:.3f}")

    # Composite health score
    health = compute_health_score(gs_dict, depth_result, edge_result, alpha_result)
    logger.info(f"  Health score: {health['score']}/100")

    # Clean up GPU memory
    for v in params.values():
        if isinstance(v, torch.Tensor):
            del v
    del params
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "health_score": health["score"],
        "health_components": health["components"],
        "health_weights": health["weights"],
        "depth_consistency": depth_result,
        "edge_alignment": edge_result,
        "alpha_coverage": alpha_result,
        "gaussian_stats": gs_dict,
    }
