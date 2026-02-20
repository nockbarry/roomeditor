"""Tests for pipeline.rasterizer module."""

import torch
import numpy as np
import pytest


@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def dummy_params(device):
    """Minimal Gaussian params for testing rasterization."""
    N = 100
    sh_degree = 2
    num_sh = (sh_degree + 1) ** 2 - 1

    return {
        "means": torch.nn.Parameter(torch.randn(N, 3, device=device) * 0.5),
        "scales": torch.nn.Parameter(torch.full((N, 3), -2.0, device=device)),
        "quats": torch.nn.Parameter(
            torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
        ),
        "opacities": torch.nn.Parameter(torch.full((N, 1), 0.5, device=device)),
        "sh0": torch.nn.Parameter(torch.randn(N, 1, 3, device=device) * 0.1),
        "shN": torch.nn.Parameter(torch.zeros(N, num_sh, 3, device=device)),
    }


@pytest.fixture
def camera(device):
    """Simple camera looking at origin."""
    K = torch.tensor(
        [[500.0, 0, 128.0], [0, 500.0, 128.0], [0, 0, 1]],
        device=device,
    ).unsqueeze(0)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    viewmat[0, 2, 3] = 3.0  # camera at z=3 looking at origin
    return K, viewmat, 256, 256


class TestRenderResult:
    def test_dataclass_fields(self):
        from pipeline.rasterizer import RenderResult
        r = RenderResult(
            image=torch.zeros(1),
            depth=torch.zeros(1),
            alphas=torch.zeros(1),
            info={},
        )
        assert r.normals is None
        assert r.surf_normals is None
        assert r.distortion is None


class TestRasterizer3DGS:
    def test_rasterize_returns_correct_shapes(self, dummy_params, camera):
        from pipeline.rasterizer import Rasterizer3DGS
        rast = Rasterizer3DGS()
        K, viewmat, W, H = camera
        result = rast.rasterize(dummy_params, viewmat, K, W, H, sh_degree=2)

        assert result.image.shape == (H, W, 3)
        assert result.depth.shape == (H, W, 1)
        assert result.alphas.shape[-1] == 1 or result.alphas.dim() >= 2
        assert isinstance(result.info, dict)
        # 3DGS should not produce normals
        assert result.normals is None
        assert result.distortion is None

    def test_rasterize_image_range(self, dummy_params, camera):
        from pipeline.rasterizer import Rasterizer3DGS
        rast = Rasterizer3DGS()
        K, viewmat, W, H = camera
        result = rast.rasterize(dummy_params, viewmat, K, W, H, sh_degree=2)

        # Rendered image should be in reasonable range (might exceed [0,1] before clamp)
        assert result.image.isfinite().all()
        assert result.depth.isfinite().all()

    def test_rasterize_has_means2d(self, dummy_params, camera):
        from pipeline.rasterizer import Rasterizer3DGS
        rast = Rasterizer3DGS()
        K, viewmat, W, H = camera
        result = rast.rasterize(dummy_params, viewmat, K, W, H, sh_degree=2)
        assert "means2d" in result.info

    def test_rasterize_preview_returns_jpeg(self, dummy_params, camera):
        from pipeline.rasterizer import Rasterizer3DGS
        rast = Rasterizer3DGS()
        K, viewmat, W, H = camera
        jpeg = rast.rasterize_preview(dummy_params, viewmat, K, W, H, sh_degree=2)

        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        # JPEG magic bytes
        assert jpeg[:2] == b'\xff\xd8'


class TestRasterizer2DGS:
    def test_rasterize_returns_correct_shapes(self, dummy_params, camera):
        from pipeline.rasterizer import Rasterizer2DGS
        rast = Rasterizer2DGS()
        K, viewmat, W, H = camera
        result = rast.rasterize(dummy_params, viewmat, K, W, H, sh_degree=2)

        assert result.image.shape == (H, W, 3)
        assert result.depth.dim() >= 2
        assert isinstance(result.info, dict)
        # 2DGS should produce normals and distortion
        assert result.normals is not None
        assert result.surf_normals is not None
        assert result.distortion is not None

    def test_rasterize_preview_returns_jpeg(self, dummy_params, camera):
        from pipeline.rasterizer import Rasterizer2DGS
        rast = Rasterizer2DGS()
        K, viewmat, W, H = camera
        jpeg = rast.rasterize_preview(dummy_params, viewmat, K, W, H, sh_degree=2)

        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        assert jpeg[:2] == b'\xff\xd8'
