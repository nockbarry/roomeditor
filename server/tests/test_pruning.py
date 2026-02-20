"""Tests for pipeline.pruning module."""

import torch
import pytest


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFloaterPruner:
    def test_init(self, device):
        from pipeline.pruning import FloaterPruner
        p = FloaterPruner(1000, device)
        assert len(p.view_counts) == 1000
        assert p.total_views == 0

    def test_resize_grow(self, device):
        from pipeline.pruning import FloaterPruner
        p = FloaterPruner(100, device)
        p.view_counts[:50] = 10
        p.resize(200)
        assert len(p.view_counts) == 200
        assert p.view_counts[25].item() == 10  # Old data preserved
        assert p.view_counts[150].item() == 0  # New entries zero

    def test_resize_shrink(self, device):
        from pipeline.pruning import FloaterPruner
        p = FloaterPruner(200, device)
        p.view_counts[:100] = 5
        p.resize(100)
        assert len(p.view_counts) == 100
        assert p.view_counts[50].item() == 5

    def test_no_prune_too_few_views(self, device):
        from pipeline.pruning import FloaterPruner
        p = FloaterPruner(100, device)
        p.total_views = 10  # Below threshold of 50
        params = {"means": torch.randn(100, 3, device=device),
                  "opacities": torch.zeros(100, 1, device=device)}
        mask = p.get_prune_mask(params)
        assert mask.sum().item() == 0

    def test_prunes_unseen_gaussians(self, device):
        from pipeline.pruning import FloaterPruner
        N = 200
        p = FloaterPruner(N, device)
        p.total_views = 100
        # First 100 Gaussians seen frequently, rest never seen
        p.view_counts[:100] = 50
        p.view_counts[100:] = 0

        params = {
            "means": torch.randn(N, 3, device=device),
            "opacities": torch.zeros(N, 1, device=device),  # Low opacity
        }
        mask = p.get_prune_mask(params, min_view_ratio=0.1)
        # Unseen Gaussians (100-199) with low opacity should be pruned
        assert mask[100:].sum().item() > 0

    def test_high_opacity_protected(self, device):
        from pipeline.pruning import FloaterPruner
        N = 100
        p = FloaterPruner(N, device)
        p.total_views = 100
        p.view_counts[:] = 0  # All unseen

        params = {
            "means": torch.randn(N, 3, device=device),
            # All high opacity (sigmoid(5) ≈ 0.99)
            "opacities": torch.full((N, 1), 5.0, device=device),
        }
        mask = p.get_prune_mask(params)
        # High-opacity Gaussians should be protected from pruning
        assert mask.sum().item() == 0

    def test_update_increments_counts(self, device):
        from pipeline.pruning import FloaterPruner
        p = FloaterPruner(100, device)
        info = {"gaussian_ids": torch.tensor([0, 1, 2, 5, 10], device=device)}
        p.update(info, step=0)
        assert p.total_views == 1
        assert p.view_counts[0].item() == 1
        assert p.view_counts[5].item() == 1
        assert p.view_counts[3].item() == 0  # Not in gaussian_ids

    def test_update_handles_missing_key(self, device):
        from pipeline.pruning import FloaterPruner
        p = FloaterPruner(100, device)
        info = {}  # No gaussian_ids
        p.update(info, step=0)
        assert p.total_views == 1
        assert p.view_counts.sum().item() == 0
