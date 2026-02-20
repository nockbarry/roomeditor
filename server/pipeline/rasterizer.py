"""Rasterizer abstraction for 3DGS and 2DGS backends."""

import io
from dataclasses import dataclass

import numpy as np
import torch
from gsplat import rasterization, rasterization_2dgs
from PIL import Image


@dataclass
class RenderResult:
    """Output from a rasterization call."""

    image: torch.Tensor  # (H, W, 3)
    depth: torch.Tensor  # (H, W, 1)
    alphas: torch.Tensor  # (H, W, 1)
    info: dict  # gsplat metadata
    normals: torch.Tensor | None = None  # (H, W, 3) — 2DGS only
    surf_normals: torch.Tensor | None = None  # (H, W, 3) — 2DGS only
    distortion: torch.Tensor | None = None  # (H, W, 1) — 2DGS only


def _ensure_batched(viewmat, K, device):
    """Convert viewmat/K to batched tensors [1, *, *] on device."""
    if not isinstance(viewmat, torch.Tensor):
        viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)
    if viewmat.dim() == 2:
        viewmat = viewmat.unsqueeze(0)
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    if K.dim() == 2:
        K = K.unsqueeze(0)
    return viewmat.to(device), K.to(device)


class Rasterizer3DGS:
    """3D Gaussian Splatting rasterizer using gsplat."""

    def rasterize(
        self,
        params: dict,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        sh_degree: int,
    ) -> RenderResult:
        renders, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"].squeeze(-1)),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
        )
        out = renders[0]  # (H, W, 4) — RGB + expected depth
        return RenderResult(
            image=out[..., :3],
            depth=out[..., 3:4],
            alphas=alphas[0],
            info=info,
        )

    def rasterize_preview(
        self,
        params: dict,
        viewmat,
        K,
        W: int,
        H: int,
        sh_degree: int,
    ) -> bytes:
        """No-grad preview render, returns JPEG bytes."""
        device = params["means"].device
        viewmat, K = _ensure_batched(viewmat, K, device)
        with torch.no_grad():
            renders, _, _ = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"].squeeze(-1)),
                colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=sh_degree,
                packed=False,
                render_mode="RGB",
            )
            preview_img = renders[0].clamp(0, 1)
            preview_np = (preview_img.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(preview_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=70)
            return buf.getvalue()


class Rasterizer2DGS:
    """2D Gaussian Splatting rasterizer using gsplat."""

    def rasterize(
        self,
        params: dict,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        sh_degree: int,
    ) -> RenderResult:
        colors, alphas, normals, surf_normals, distort, median_depth, info = (
            rasterization_2dgs(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"].squeeze(-1)),
                colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=sh_degree,
                packed=False,
                render_mode="RGB+ED",
                distloss=True,
            )
        )
        return RenderResult(
            image=colors[0, ..., :3],
            depth=median_depth[0],
            alphas=alphas[0],
            info=info,
            normals=normals[0],
            surf_normals=surf_normals[0],
            distortion=distort[0],
        )

    def rasterize_preview(
        self,
        params: dict,
        viewmat,
        K,
        W: int,
        H: int,
        sh_degree: int,
    ) -> bytes:
        """No-grad preview render, returns JPEG bytes."""
        device = params["means"].device
        viewmat, K = _ensure_batched(viewmat, K, device)
        with torch.no_grad():
            colors, *_ = rasterization_2dgs(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"].squeeze(-1)),
                colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=sh_degree,
                packed=False,
                render_mode="RGB",
                distloss=False,
            )
            preview_img = colors[0, ..., :3].clamp(0, 1)
            preview_np = (preview_img.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(preview_np)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=70)
            return buf.getvalue()
