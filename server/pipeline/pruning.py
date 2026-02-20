"""TIDI-GS-inspired floater pruning for Gaussian Splatting.

Tracks per-Gaussian visibility across training views and prunes
spatially isolated, low-evidence Gaussians that are likely floaters.
"""

import torch


class FloaterPruner:
    """Multi-view floater detection and pruning."""

    def __init__(self, n_gaussians: int, device: torch.device):
        self.view_counts = torch.zeros(n_gaussians, device=device)
        self.total_views = 0

    def update(self, info: dict, step: int):
        """Called each training step with rasterization info."""
        if "gaussian_ids" in info and info["gaussian_ids"] is not None:
            visible_ids = info["gaussian_ids"].unique()
            valid = visible_ids < len(self.view_counts)
            self.view_counts[visible_ids[valid]] += 1
        elif "radii" in info and info["radii"] is not None:
            # Unpacked mode: radii shape (B, N, 2), visible if any radius > 0
            radii = info["radii"]
            if radii.dim() == 3:
                visible = (radii[0].max(dim=-1).values > 0).nonzero(as_tuple=True)[0]
            else:
                visible = (radii.max(dim=-1).values > 0).nonzero(as_tuple=True)[0]
            valid = visible < len(self.view_counts)
            self.view_counts[visible[valid]] += 1
        self.total_views += 1

    def get_prune_mask(
        self, params: dict, min_view_ratio: float = 0.1
    ) -> torch.Tensor:
        """Return boolean mask of Gaussians to prune."""
        n = len(params["means"])
        device = params["means"].device
        mask = torch.zeros(n, dtype=torch.bool, device=device)

        if self.total_views < 50:
            return mask  # Not enough data yet

        # 1. Evidence-aware: prune if seen in < min_view_ratio of views
        counts = self.view_counts[:n]
        view_ratio = counts / max(self.total_views, 1)
        mask |= view_ratio < min_view_ratio

        # 2. Context-aware: prune if spatially isolated
        # Use batched KNN against a small random anchor set to stay O(N*K)
        means = params["means"].detach()
        anchor_size = min(n, 4096)
        anchor_idx = torch.randperm(n, device=device)[:anchor_size]
        anchors = means[anchor_idx]  # (K, 3)

        # Compute distances in batches to limit memory
        batch_size = 8192
        nearest = torch.full((n,), float("inf"), device=device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = means[start:end]  # (B, 3)
            dists = torch.cdist(chunk, anchors)  # (B, K)
            # Exclude self-matches (distance ~0)
            dists[dists < 1e-8] = float("inf")
            nearest[start:end] = dists.min(dim=1).values

        median_dist = nearest[nearest.isfinite()].median()
        mask |= nearest > 5 * median_dist

        # Don't prune high-opacity Gaussians (they're likely real)
        high_opacity = (
            torch.sigmoid(params["opacities"].squeeze(-1)) > 0.8
        )
        mask &= ~high_opacity

        return mask

    def resize(self, new_n: int):
        """Called after densification changes Gaussian count."""
        old_n = len(self.view_counts)
        if new_n > old_n:
            self.view_counts = torch.cat([
                self.view_counts,
                torch.zeros(
                    new_n - old_n, device=self.view_counts.device
                ),
            ])
        elif new_n < old_n:
            self.view_counts = self.view_counts[:new_n]
