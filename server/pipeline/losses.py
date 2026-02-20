"""Loss functions for Gaussian Splatting training."""

import torch
import torch.nn.functional as F


def l1_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 photometric loss between rendered and ground truth images."""
    return F.l1_loss(rendered, gt)


def ssim_loss(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """Compute 1 - SSIM between two (H, W, 3) images. Returns scalar."""
    # Convert to (1, 3, H, W)
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)
    window = window.expand(3, 1, -1, -1)  # (3, 1, ws, ws)

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
    return 1.0 - ssim_map.mean()


def depth_tv_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware depth total variation loss.

    Penalizes depth discontinuities except where the RGB image also has edges,
    which encourages smooth surfaces while preserving object boundaries.

    Args:
        depth: (H, W, 1) expected depth map
        image: (H, W, 3) ground truth RGB image
    """
    depth_dx = torch.abs(depth[1:, :, :] - depth[:-1, :, :])
    depth_dy = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])

    img_dx = torch.mean(torch.abs(image[1:, :, :] - image[:-1, :, :]), dim=-1, keepdim=True)
    img_dy = torch.mean(torch.abs(image[:, 1:, :] - image[:, :-1, :]), dim=-1, keepdim=True)

    depth_dx = depth_dx * torch.exp(-img_dx * 10.0)
    depth_dy = depth_dy * torch.exp(-img_dy * 10.0)

    return depth_dx.mean() + depth_dy.mean()


def opacity_reg_loss(opacities: torch.Tensor) -> torch.Tensor:
    """Binary entropy loss on opacities.

    Pushes opacities toward 0 or 1, penalizing semi-transparent floaters.
    opacities: (N,) values in [0,1] (already sigmoid-activated).
    """
    o = opacities.clamp(1e-6, 1 - 1e-6)
    return -(o * torch.log(o) + (1 - o) * torch.log(1 - o)).mean()


def scale_reg_loss(scales: torch.Tensor) -> torch.Tensor:
    """Penalize oversized blob Gaussians.

    scales: (N, 3) in log-space (before exp activation).
    """
    return torch.exp(scales).mean()


def flatten_reg_loss(scales: torch.Tensor) -> torch.Tensor:
    """Encourage disk-shaped Gaussians flat against surfaces.

    Penalizes the smallest scale axis, pushing Gaussians toward 2D disks.
    scales: (N, 3) in log-space (before exp activation).
    """
    min_scale = scales.min(dim=-1).values  # (N,)
    return torch.exp(min_scale).mean()


def normal_consistency_loss(
    normals: torch.Tensor, surf_normals: torch.Tensor
) -> torch.Tensor:
    """Penalize disagreement between rendered and surface normals (2DGS only)."""
    return (1 - (normals * surf_normals).sum(dim=-1)).mean()


def distortion_loss(distort_map: torch.Tensor) -> torch.Tensor:
    """L1 distortion from 2DGS paper — already computed by rasterizer."""
    return distort_map.mean()
