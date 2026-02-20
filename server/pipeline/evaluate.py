"""Evaluation metrics for Gaussian Splatting models.

Provides PSNR, SSIM, and LPIPS metrics on held-out test views,
following the standard Mip-NeRF 360 evaluation protocol.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def split_train_test(
    n_images: int, test_every: int = 8
) -> tuple[list[int], list[int]]:
    """Split image indices into train/test sets.

    Holds out every `test_every`-th image for testing,
    matching the standard Mip-NeRF 360 evaluation convention.

    Args:
        n_images: Total number of images.
        test_every: Hold out every N-th image (default 8).

    Returns:
        (train_indices, test_indices) — disjoint sorted lists.
    """
    train_indices = []
    test_indices = []
    for i in range(n_images):
        if i % test_every == 0:
            test_indices.append(i)
        else:
            train_indices.append(i)
    return train_indices, test_indices


def compute_psnr(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        rendered: (H, W, 3) float32 tensor in [0, 1].
        gt: (H, W, 3) float32 tensor in [0, 1].

    Returns:
        PSNR in dB.
    """
    mse = F.mse_loss(rendered, gt).item()
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def compute_ssim(
    rendered: torch.Tensor, gt: torch.Tensor, window_size: int = 11
) -> float:
    """Compute Structural Similarity Index.

    Uses the same Gaussian-weighted SSIM kernel as the training loss.

    Args:
        rendered: (H, W, 3) float32 tensor in [0, 1].
        gt: (H, W, 3) float32 tensor in [0, 1].
        window_size: Size of the Gaussian window (default 11).

    Returns:
        SSIM value in [0, 1] (higher is better).
    """
    x = rendered.permute(2, 0, 1).unsqueeze(0)
    y = gt.permute(2, 0, 1).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    coords = torch.arange(window_size, dtype=torch.float32, device=x.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(x, window, padding=pad, groups=3)
    mu2 = F.conv2d(y, window, padding=pad, groups=3)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(x * x, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def compute_lpips(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lpips_net: "lpips.LPIPS",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity.

    Args:
        rendered: (H, W, 3) float32 tensor in [0, 1].
        gt: (H, W, 3) float32 tensor in [0, 1].
        lpips_net: Pre-initialized lpips.LPIPS network.

    Returns:
        LPIPS distance (lower is better).
    """
    # LPIPS expects (N, 3, H, W) in [-1, 1]
    x = rendered.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    y = gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return lpips_net(x, y).item()


def _init_lpips(device: torch.device) -> "lpips.LPIPS":
    """Lazily initialize LPIPS VGG network."""
    import lpips

    net = lpips.LPIPS(net="vgg").to(device)
    net.eval()
    return net


@dataclass
class EvalResults:
    """Aggregated evaluation results across test views."""

    per_view: list[dict] = field(default_factory=list)
    mean_psnr: float = 0.0
    mean_ssim: float = 0.0
    mean_lpips: float = 0.0
    num_test_views: int = 0
    num_gaussians: int = 0

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def save_json(self, path: Path):
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "EvalResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def print_table(self):
        """Print a formatted results table."""
        print(f"\nEvaluation Results ({self.num_test_views} test views, "
              f"{self.num_gaussians:,} Gaussians)")
        print("=" * 60)
        print(f"{'View':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        print("-" * 60)
        for v in self.per_view:
            print(f"{v['name']:<30} {v['psnr']:>8.2f} {v['ssim']:>8.4f} "
                  f"{v['lpips']:>8.4f}")
        print("-" * 60)
        print(f"{'Mean':<30} {self.mean_psnr:>8.2f} {self.mean_ssim:>8.4f} "
              f"{self.mean_lpips:>8.4f}")
        print()


def evaluate_model(
    trainer,
    test_indices: list[int],
    all_cameras: list[dict],
    all_images: list[torch.Tensor],
    save_dir: Path | None = None,
) -> EvalResults:
    """Evaluate a trained model on held-out test views.

    Args:
        trainer: GaussianTrainer with trained params.
        test_indices: Indices of test views in the full camera/image lists.
        all_cameras: Full list of camera dicts (before train/test split).
        all_images: Full list of GT image tensors (before train/test split).
        save_dir: If provided, save rendered test images as PNGs.

    Returns:
        EvalResults with per-view and mean metrics.
    """
    device = trainer.device
    lpips_net = _init_lpips(device)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    per_view = []
    psnrs, ssims, lpipss = [], [], []

    for i, test_idx in enumerate(test_indices):
        cam = all_cameras[test_idx]
        gt_image = all_images[test_idx]
        W, H = cam["width"], cam["height"]

        viewmat = torch.tensor(cam["viewmat"], device=device).unsqueeze(0)
        K = torch.tensor(cam["K"], device=device).unsqueeze(0)

        with torch.no_grad():
            result = trainer.rasterizer.rasterize(
                trainer.params, viewmat, K, W, H, trainer.config.sh_degree
            )
            rendered = result.image.clamp(0, 1)

        psnr = compute_psnr(rendered, gt_image)
        ssim = compute_ssim(rendered, gt_image)
        lpips_val = compute_lpips(rendered, gt_image, lpips_net)

        img_name = Path(cam.get("image_path", f"view_{test_idx:04d}")).stem
        per_view.append({
            "name": img_name,
            "index": test_idx,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "lpips": round(lpips_val, 4),
        })
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips_val)

        if save_dir is not None:
            rendered_np = (rendered.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(rendered_np).save(save_dir / f"{img_name}.png")

        if (i + 1) % 10 == 0 or (i + 1) == len(test_indices):
            logger.info(
                f"  Evaluated {i + 1}/{len(test_indices)} test views "
                f"(PSNR={psnr:.2f})"
            )

    n_gaussians = len(trainer.params["means"])

    results = EvalResults(
        per_view=per_view,
        mean_psnr=round(float(np.mean(psnrs)), 4),
        mean_ssim=round(float(np.mean(ssims)), 4),
        mean_lpips=round(float(np.mean(lpipss)), 4),
        num_test_views=len(test_indices),
        num_gaussians=n_gaussians,
    )

    del lpips_net
    torch.cuda.empty_cache()

    return results
