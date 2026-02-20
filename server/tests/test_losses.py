"""Tests for pipeline.losses module."""

import torch
import pytest


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_images(device):
    """Two random (H, W, 3) images."""
    img1 = torch.rand(64, 64, 3, device=device)
    img2 = torch.rand(64, 64, 3, device=device)
    return img1, img2


class TestL1Loss:
    def test_zero_for_identical(self, device):
        from pipeline.losses import l1_loss
        img = torch.rand(32, 32, 3, device=device)
        assert l1_loss(img, img).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different(self, random_images):
        from pipeline.losses import l1_loss
        img1, img2 = random_images
        assert l1_loss(img1, img2).item() > 0

    def test_symmetric(self, random_images):
        from pipeline.losses import l1_loss
        img1, img2 = random_images
        assert l1_loss(img1, img2).item() == pytest.approx(
            l1_loss(img2, img1).item(), abs=1e-6
        )


class TestSSIMLoss:
    def test_zero_for_identical(self, device):
        from pipeline.losses import ssim_loss
        img = torch.rand(64, 64, 3, device=device)
        # SSIM of identical images = 1.0, so 1 - SSIM = 0.0
        loss = ssim_loss(img, img)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_for_different(self, random_images):
        from pipeline.losses import ssim_loss
        img1, img2 = random_images
        loss = ssim_loss(img1, img2)
        assert 0.0 < loss.item() <= 2.0

    def test_returns_scalar(self, random_images):
        from pipeline.losses import ssim_loss
        img1, img2 = random_images
        loss = ssim_loss(img1, img2)
        assert loss.dim() == 0


class TestDepthTVLoss:
    def test_zero_for_constant_depth(self, device):
        from pipeline.losses import depth_tv_loss
        depth = torch.ones(64, 64, 1, device=device) * 5.0
        image = torch.rand(64, 64, 3, device=device)
        loss = depth_tv_loss(depth, image)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_varying_depth(self, device):
        from pipeline.losses import depth_tv_loss
        depth = torch.rand(64, 64, 1, device=device)
        image = torch.zeros(64, 64, 3, device=device)  # No edges → full penalty
        loss = depth_tv_loss(depth, image)
        assert loss.item() > 0

    def test_edges_reduce_penalty(self, device):
        from pipeline.losses import depth_tv_loss
        depth = torch.rand(64, 64, 1, device=device)
        image_flat = torch.zeros(64, 64, 3, device=device)
        image_edgy = torch.rand(64, 64, 3, device=device)
        loss_flat = depth_tv_loss(depth, image_flat)
        loss_edgy = depth_tv_loss(depth, image_edgy)
        # Edges in image should reduce depth penalty
        assert loss_edgy.item() <= loss_flat.item()


class TestOpacityRegLoss:
    def test_min_at_extremes(self, device):
        from pipeline.losses import opacity_reg_loss
        # Opacities near 0 and 1 → low entropy
        extreme = torch.tensor([0.01, 0.99, 0.01, 0.99], device=device)
        mid = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device)
        loss_extreme = opacity_reg_loss(extreme)
        loss_mid = opacity_reg_loss(mid)
        assert loss_extreme.item() < loss_mid.item()

    def test_returns_positive(self, device):
        from pipeline.losses import opacity_reg_loss
        o = torch.rand(100, device=device)
        assert opacity_reg_loss(o).item() > 0


class TestScaleRegLoss:
    def test_larger_scales_higher_loss(self, device):
        from pipeline.losses import scale_reg_loss
        small = torch.full((100, 3), -3.0, device=device)  # exp(-3) ≈ 0.05
        large = torch.full((100, 3), 2.0, device=device)   # exp(2) ≈ 7.4
        assert scale_reg_loss(small).item() < scale_reg_loss(large).item()


class TestFlattenRegLoss:
    def test_flat_gaussians_lower_loss(self, device):
        from pipeline.losses import flatten_reg_loss
        # Flat: one axis much smaller
        flat = torch.tensor([[1.0, 1.0, -5.0]] * 100, device=device)
        # Round: all axes equal
        round_ = torch.tensor([[1.0, 1.0, 1.0]] * 100, device=device)
        assert flatten_reg_loss(flat).item() < flatten_reg_loss(round_).item()


class TestNormalConsistencyLoss:
    def test_zero_for_aligned(self, device):
        from pipeline.losses import normal_consistency_loss
        n = torch.randn(32, 32, 3, device=device)
        n = n / n.norm(dim=-1, keepdim=True)
        loss = normal_consistency_loss(n, n)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_misaligned(self, device):
        from pipeline.losses import normal_consistency_loss
        n1 = torch.randn(32, 32, 3, device=device)
        n1 = n1 / n1.norm(dim=-1, keepdim=True)
        n2 = torch.randn(32, 32, 3, device=device)
        n2 = n2 / n2.norm(dim=-1, keepdim=True)
        loss = normal_consistency_loss(n1, n2)
        assert loss.item() > 0


class TestDistortionLoss:
    def test_returns_mean(self, device):
        from pipeline.losses import distortion_loss
        d = torch.rand(32, 32, 1, device=device) * 0.1
        loss = distortion_loss(d)
        assert loss.item() == pytest.approx(d.mean().item(), abs=1e-6)

    def test_zero_for_zero_input(self, device):
        from pipeline.losses import distortion_loss
        d = torch.zeros(32, 32, 1, device=device)
        assert distortion_loss(d).item() == pytest.approx(0.0, abs=1e-6)
