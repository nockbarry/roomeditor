"""No-reference quality evaluation for 3D Gaussian Splatting reconstructions.

Computes metrics to compare reconstruction quality WITHOUT ground-truth,
using self-consistency, geometric plausibility, and perceptual quality.

Metrics:
    1. Training-view PSNR/SSIM/LPIPS (self-consistency)
    2. Gaussian opacity histogram / floater fraction
    3. Gaussian scale distribution stats
    4. CLIP-IQA (no-reference perceptual quality)
    5. NIQE (natural image quality)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GaussianStats:
    """Statistics about Gaussian primitive distribution."""
    n_gaussians: int = 0
    n_effective: int = 0  # opacity > 0.1
    frac_transparent: float = 0.0  # opacity < 0.05
    frac_opaque: float = 0.0  # opacity > 0.95
    opacity_mean: float = 0.0
    opacity_std: float = 0.0
    log_scale_mean: float = 0.0
    log_scale_std: float = 0.0
    log_scale_min: float = 0.0
    log_scale_max: float = 0.0
    frac_scale_outlier: float = 0.0  # > 3 std from mean
    mean_nn_dist: float = 0.0  # mean nearest-neighbor distance
    bbox_volume: float = 0.0  # axis-aligned bounding box volume
    density: float = 0.0  # effective Gaussians per unit volume


@dataclass
class NoRefMetrics:
    """No-reference evaluation results."""
    # Self-consistency (training-view re-rendering)
    train_psnr: float = 0.0
    train_ssim: float = 0.0
    train_lpips: float = 0.0
    # NR-IQA on rendered views
    clip_iqa: float = 0.0
    niqe: float = 0.0
    # Gaussian primitive stats
    gaussian_stats: GaussianStats = field(default_factory=GaussianStats)

    def as_dict(self) -> dict:
        d = {
            "train_psnr": round(self.train_psnr, 2),
            "train_ssim": round(self.train_ssim, 4),
            "train_lpips": round(self.train_lpips, 4),
            "clip_iqa": round(self.clip_iqa, 4),
            "niqe": round(self.niqe, 4),
        }
        gs = self.gaussian_stats
        d["gaussian_stats"] = {
            "n_gaussians": gs.n_gaussians,
            "n_effective": gs.n_effective,
            "frac_transparent": round(gs.frac_transparent, 4),
            "frac_opaque": round(gs.frac_opaque, 4),
            "opacity_mean": round(gs.opacity_mean, 4),
            "log_scale_mean": round(gs.log_scale_mean, 4),
            "log_scale_std": round(gs.log_scale_std, 4),
            "frac_scale_outlier": round(gs.frac_scale_outlier, 4),
            "mean_nn_dist": round(gs.mean_nn_dist, 6),
            "bbox_volume": round(gs.bbox_volume, 4),
            "density": round(gs.density, 2),
        }
        return d

    def print_table(self):
        print(f"\n{'No-Reference Evaluation':}")
        print("=" * 50)
        print(f"  Training-view PSNR:  {self.train_psnr:.2f} dB")
        print(f"  Training-view SSIM:  {self.train_ssim:.4f}")
        print(f"  Training-view LPIPS: {self.train_lpips:.4f}")
        print(f"  CLIP-IQA:            {self.clip_iqa:.4f}")
        print(f"  NIQE:                {self.niqe:.4f}")
        gs = self.gaussian_stats
        print(f"\n{'Gaussian Primitive Stats':}")
        print(f"  Total:       {gs.n_gaussians:,}")
        print(f"  Effective:   {gs.n_effective:,} (opacity > 0.1)")
        print(f"  Transparent: {gs.frac_transparent:.1%} (opacity < 0.05)")
        print(f"  Opaque:      {gs.frac_opaque:.1%} (opacity > 0.95)")
        print(f"  Log-scale:   {gs.log_scale_mean:.2f} ± {gs.log_scale_std:.2f}")
        print(f"  Scale outliers: {gs.frac_scale_outlier:.1%}")
        print(f"  Density:     {gs.density:.1f} Gaussians/m³")


def compute_gaussian_stats(ply_path: Path) -> GaussianStats:
    """Analyze Gaussian primitive distribution from a PLY file."""
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    stats = GaussianStats()

    # Extract positions
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    stats.n_gaussians = len(means)

    if stats.n_gaussians == 0:
        return stats

    # Extract opacities (stored as logits in standard 3DGS PLY)
    if "opacity" in vertex.data.dtype.names:
        opa_logit = np.array(vertex["opacity"])
        opacities = 1.0 / (1.0 + np.exp(-opa_logit))
    else:
        opacities = np.ones(stats.n_gaussians) * 0.5

    stats.n_effective = int((opacities > 0.1).sum())
    stats.frac_transparent = float((opacities < 0.05).mean())
    stats.frac_opaque = float((opacities > 0.95).mean())
    stats.opacity_mean = float(opacities.mean())
    stats.opacity_std = float(opacities.std())

    # Extract scales (stored as log-scales in standard 3DGS PLY)
    if "scale_0" in vertex.data.dtype.names:
        log_scales = np.stack([
            vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]
        ], axis=-1)
        stats.log_scale_mean = float(log_scales.mean())
        stats.log_scale_std = float(log_scales.std())
        stats.log_scale_min = float(log_scales.min())
        stats.log_scale_max = float(log_scales.max())

        # Outliers: more than 3 std from mean
        outlier_mask = np.abs(log_scales - log_scales.mean()) > 3 * log_scales.std()
        stats.frac_scale_outlier = float(outlier_mask.any(axis=-1).mean())

    # Spatial stats
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    bbox_extent = bbox_max - bbox_min
    stats.bbox_volume = float(np.prod(bbox_extent))

    if stats.bbox_volume > 0 and stats.n_effective > 0:
        stats.density = stats.n_effective / stats.bbox_volume

    # Nearest-neighbor distance (subsample for speed)
    from scipy.spatial import KDTree
    n_sample = min(10000, len(means))
    idx = np.random.choice(len(means), n_sample, replace=False)
    tree = KDTree(means[idx])
    dists, _ = tree.query(means[idx], k=2)  # k=2 because closest is self
    stats.mean_nn_dist = float(dists[:, 1].mean())

    return stats


def compute_nriqa(rendered_images: list[torch.Tensor]) -> dict:
    """Compute no-reference image quality metrics on rendered views.

    Args:
        rendered_images: list of (H, W, 3) float tensors [0, 1] on GPU.

    Returns:
        dict with clip_iqa and niqe scores (averaged over all views).
    """
    results = {"clip_iqa": 0.0, "niqe": 0.0}

    if not rendered_images:
        return results

    try:
        import pyiqa
    except ImportError:
        logger.warning("pyiqa not installed, skipping NR-IQA metrics")
        return results

    device = rendered_images[0].device

    # Stack and convert to NCHW format
    batch = torch.stack([img.permute(2, 0, 1) for img in rendered_images])

    try:
        clipiqa = pyiqa.create_metric("clipiqa", device=device)
        scores = clipiqa(batch)
        results["clip_iqa"] = float(scores.mean())
        del clipiqa
    except Exception as e:
        logger.warning(f"CLIP-IQA failed: {e}")

    try:
        niqe = pyiqa.create_metric("niqe", device=device)
        scores = niqe(batch)
        results["niqe"] = float(scores.mean())
        del niqe
    except Exception as e:
        logger.warning(f"NIQE failed: {e}")

    torch.cuda.empty_cache()
    return results


def compute_self_consistency(
    rendered_images: list[torch.Tensor],
    gt_images: list[torch.Tensor],
) -> dict:
    """Compute training-view PSNR/SSIM/LPIPS (self-consistency).

    Args:
        rendered_images: list of (H, W, 3) tensors [0, 1]
        gt_images: list of (H, W, 3) tensors [0, 1] (input training views)

    Returns:
        dict with train_psnr, train_ssim, train_lpips
    """
    import lpips as lpips_lib

    psnrs = []
    ssims = []
    lpipss = []

    lpips_fn = lpips_lib.LPIPS(net="vgg").cuda()

    for rendered, gt in zip(rendered_images, gt_images):
        rendered = rendered.clamp(0, 1)
        gt = gt.clamp(0, 1)

        # PSNR
        mse = ((rendered - gt) ** 2).mean()
        psnr = -10 * torch.log10(mse + 1e-8)
        psnrs.append(psnr.item())

        # SSIM (from torchmetrics)
        from torchmetrics.functional.image import structural_similarity_index_measure
        r_nchw = rendered.permute(2, 0, 1).unsqueeze(0)
        g_nchw = gt.permute(2, 0, 1).unsqueeze(0)
        ssim_val = structural_similarity_index_measure(r_nchw, g_nchw, data_range=1.0)
        ssims.append(ssim_val.item())

        # LPIPS
        # Convert to [-1, 1] NCHW
        r_lpips = r_nchw * 2 - 1
        g_lpips = g_nchw * 2 - 1
        lpips_val = lpips_fn(r_lpips, g_lpips)
        lpipss.append(lpips_val.item())

    del lpips_fn
    torch.cuda.empty_cache()

    return {
        "train_psnr": float(np.mean(psnrs)),
        "train_ssim": float(np.mean(ssims)),
        "train_lpips": float(np.mean(lpipss)),
    }
