"""3D/2D Gaussian Splatting training using gsplat library.

Reads a COLMAP sparse reconstruction, initializes Gaussians from the
sparse point cloud, then optimizes them to match the training images.
"""

import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, export_splats
from PIL import Image

from pipeline.losses import (
    l1_loss,
    ssim_loss,
    depth_tv_loss,
    opacity_reg_loss,
    scale_reg_loss,
    flatten_reg_loss,
    normal_consistency_loss,
    distortion_loss,
)
from pipeline.rasterizer import Rasterizer3DGS, Rasterizer2DGS

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    iterations: int = 15_000
    sh_degree: int = 2
    mode: str = "3dgs"  # "3dgs" or "2dgs"
    depth_reg_weight: float = 0.1
    opacity_reg_weight: float = 0.0
    scale_reg_weight: float = 0.0
    flatten_reg_weight: float = 0.0
    distortion_weight: float = 0.0
    normal_weight: float = 0.0
    sh_degree_interval: int = 1000
    prune_opa: float = 0.005
    densify_until_pct: float = 0.5
    appearance_embeddings: bool = False
    appearance_embed_dim: int = 32
    appearance_reg_weight: float = 0.1
    tidi_pruning: bool = False


@dataclass
class TrainerCallbacks:
    progress: Callable[[float], None] | None = None
    preview: Callable[[int, int, float, int, bytes], None] | None = None
    snapshot: Callable[[int, int, float, int, str], None] | None = None
    preview_every: int = 500
    snapshot_every: int = 2000


def load_colmap_data(
    model_dir: Path, images_dir: Path
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load COLMAP reconstruction and training images.

    Returns:
        cameras_data: list of dicts with keys: image_path, K (3x3), viewmat (4x4), width, height
        points3d: (N, 3) float array of sparse 3D points
        colors3d: (N, 3) float array of point colors [0,1]
    """
    recon = pycolmap.Reconstruction(str(model_dir))

    pts = []
    cols = []
    for pid, p3d in recon.points3D.items():
        pts.append(p3d.xyz)
        cols.append(np.array(p3d.color, dtype=np.float64) / 255.0)

    if not pts:
        raise RuntimeError("COLMAP reconstruction has no 3D points")

    points3d = np.array(pts, dtype=np.float32)
    colors3d = np.array(cols, dtype=np.float32)

    cameras_data = []
    for img_id in recon.reg_image_ids():
        image = recon.image(img_id)
        camera = recon.camera(image.camera_id)

        img_path = images_dir / image.name
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}, skipping")
            continue

        fx, fy, cx, cy = (
            camera.focal_length_x,
            camera.focal_length_y,
            camera.principal_point_x,
            camera.principal_point_y,
        )
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        cameras_data.append({
            "image_path": str(img_path),
            "K": K,
            "viewmat": viewmat,
            "width": camera.width,
            "height": camera.height,
        })

    if not cameras_data:
        raise RuntimeError("No registered images found in COLMAP reconstruction")

    logger.info(
        f"Loaded {len(cameras_data)} images and {len(points3d)} 3D points "
        f"from COLMAP"
    )
    return cameras_data, points3d, colors3d


def _load_image(path: str, width: int, height: int) -> torch.Tensor:
    """Load an image and return as (H, W, 3) float32 tensor on GPU [0,1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device="cuda")


class GaussianTrainer:
    """Modular Gaussian Splatting trainer."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda")
        self.params: dict[str, torch.nn.Parameter] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.strategy: DefaultStrategy | None = None
        self.strategy_state: dict | None = None
        self.cameras_data: list[dict] = []
        self.gt_images: list[torch.Tensor] = []
        self.current_step: int = 0

        # Appearance model (initialized in init_params if enabled)
        self.appearance_embeds: torch.nn.Embedding | None = None
        self.appearance_mlp: torch.nn.Sequential | None = None

        # Pruner (initialized in init_strategy if enabled)
        self.pruner = None

        # Select rasterizer based on mode
        if config.mode == "2dgs":
            self.rasterizer = Rasterizer2DGS()
        else:
            self.rasterizer = Rasterizer3DGS()

    def load_data(self, colmap_dir: Path, images_dir: Path):
        """Load COLMAP reconstruction and training images."""
        self.cameras_data, points3d, colors3d = load_colmap_data(
            colmap_dir, images_dir
        )

        # Downscale images for training efficiency
        downscale = 1
        if self.cameras_data[0]["width"] > 1600:
            downscale = 2
        if self.cameras_data[0]["width"] > 3200:
            downscale = 4

        if downscale > 1:
            for cam in self.cameras_data:
                cam["width"] //= downscale
                cam["height"] //= downscale
                cam["K"][:2] /= downscale
            logger.info(f"Downscaling images by {downscale}x")

        # Pre-load all images to GPU
        logger.info("Loading training images to GPU...")
        self.gt_images = []
        for cam in self.cameras_data:
            self.gt_images.append(
                _load_image(cam["image_path"], cam["width"], cam["height"])
            )

        return points3d, colors3d

    def set_training_views(self, indices: list[int]):
        """Restrict training to only the given camera indices.

        Call after load_data() but before train() to hold out test views.
        """
        self.cameras_data = [self.cameras_data[i] for i in indices]
        self.gt_images = [self.gt_images[i] for i in indices]

    def init_params(self, points3d: np.ndarray, colors3d: np.ndarray):
        """Initialize Gaussian parameters from sparse point cloud."""
        N = len(points3d)
        sh_degree = self.config.sh_degree

        means = torch.tensor(points3d, dtype=torch.float32, device=self.device)

        # Compute initial scale from local point density
        from torch import cdist

        if N > 10000:
            sample_idx = torch.randperm(N)[:10000]
            dists = cdist(means[sample_idx], means[sample_idx])
        else:
            dists = cdist(means, means)
        dists[dists == 0] = float("inf")
        knn_dists = dists.topk(3, dim=1, largest=False).values
        avg_dist = knn_dists.mean(dim=1)

        if N > 10000:
            global_avg_dist = avg_dist.mean().item()
            scales = torch.full(
                (N, 3), math.log(global_avg_dist), device=self.device
            )
        else:
            scales = torch.log(
                avg_dist.unsqueeze(1).expand(-1, 3).clamp(min=1e-6)
            )

        quats = torch.zeros(N, 4, device=self.device)
        quats[:, 0] = 1.0

        opacities = torch.full((N, 1), 0.5, device=self.device)

        C0 = 0.28209479177387814
        sh0 = (
            torch.tensor(colors3d, dtype=torch.float32, device=self.device) - 0.5
        ) / C0
        sh0 = sh0.unsqueeze(1)

        num_sh_extra = (sh_degree + 1) ** 2 - 1
        shN = torch.zeros(N, num_sh_extra, 3, device=self.device)

        self.params = {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }

        self._init_optimizers()

        # Initialize appearance model if enabled
        if self.config.appearance_embeddings:
            self._init_appearance_model()

        logger.info(f"Initialized {N} Gaussians from sparse points")

    def _init_optimizers(self):
        """Set up optimizers for Gaussian parameters."""
        self.optimizers = {
            "means": torch.optim.Adam([self.params["means"]], lr=1.6e-4),
            "scales": torch.optim.Adam([self.params["scales"]], lr=5e-3),
            "quats": torch.optim.Adam([self.params["quats"]], lr=1e-3),
            "opacities": torch.optim.Adam([self.params["opacities"]], lr=5e-2),
            "sh0": torch.optim.Adam([self.params["sh0"]], lr=2.5e-3),
            "shN": torch.optim.Adam([self.params["shN"]], lr=2.5e-3),
        }

    def _init_appearance_model(self):
        """Initialize per-image appearance embeddings + affine color MLP."""
        n_images = len(self.cameras_data)
        embed_dim = self.config.appearance_embed_dim

        self.appearance_embeds = torch.nn.Embedding(n_images, embed_dim).to(
            self.device
        )
        self.appearance_mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 12),  # 3x3 color matrix + 3 bias = 12
        ).to(self.device)

        # Initialize to identity transform
        torch.nn.init.zeros_(self.appearance_mlp[-1].weight)
        torch.nn.init.zeros_(self.appearance_mlp[-1].bias)
        self.appearance_mlp[-1].bias.data[:9] = torch.eye(3).flatten()

        self.optimizers["appearance_embeds"] = torch.optim.Adam(
            self.appearance_embeds.parameters(), lr=1e-4
        )
        self.optimizers["appearance_mlp"] = torch.optim.Adam(
            self.appearance_mlp.parameters(), lr=1e-4
        )

    def init_strategy(self):
        """Set up densification strategy and optional pruner."""
        refine_stop = int(self.config.iterations * self.config.densify_until_pct)

        strategy_kwargs = dict(
            prune_opa=self.config.prune_opa,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            refine_start_iter=500,
            refine_stop_iter=refine_stop,
            reset_every=3000,
            refine_every=100,
            verbose=False,
        )

        if self.config.mode == "2dgs":
            strategy_kwargs["key_for_gradient"] = "gradient_2dgs"

        self.strategy = DefaultStrategy(**strategy_kwargs)
        self.strategy_state = self.strategy.initialize_state()

        # Initialize pruner if enabled
        if self.config.tidi_pruning:
            from pipeline.pruning import FloaterPruner

            n_gaussians = len(self.params["means"])
            self.pruner = FloaterPruner(n_gaussians, self.device)

    def _apply_appearance(
        self, rendered_image: torch.Tensor, image_idx: int
    ) -> torch.Tensor:
        """Apply per-image appearance transform during training."""
        embed = self.appearance_embeds(
            torch.tensor(image_idx, device=self.device)
        )
        affine = self.appearance_mlp(embed)  # (12,)
        color_matrix = affine[:9].reshape(3, 3)
        color_bias = affine[9:]
        rendered_image = (rendered_image @ color_matrix.T) + color_bias
        return rendered_image.clamp(0, 1)

    def train_step(self, step: int) -> dict:
        """Single training iteration. Returns loss dict."""
        config = self.config
        num_views = len(self.cameras_data)

        # Pick a random training view
        idx = torch.randint(0, num_views, (1,)).item()
        cam = self.cameras_data[idx]
        gt_image = self.gt_images[idx]
        W, H = cam["width"], cam["height"]

        viewmat = torch.tensor(cam["viewmat"], device=self.device).unsqueeze(0)
        K = torch.tensor(cam["K"], device=self.device).unsqueeze(0)

        # SH degree ramping: start with DC only, increase every sh_degree_interval steps
        active_sh = min(config.sh_degree, step // config.sh_degree_interval)

        # Forward: rasterize
        result = self.rasterizer.rasterize(
            self.params, viewmat, K, W, H, active_sh
        )

        # Retain gradient for densification
        if "means2d" in result.info:
            result.info["means2d"].retain_grad()

        rendered_image = result.image

        # Apply appearance transform if enabled (training only)
        if config.appearance_embeddings and self.appearance_embeds is not None:
            rendered_image = self._apply_appearance(rendered_image, idx)

        # Compute losses
        loss_l1 = l1_loss(rendered_image, gt_image)
        loss_ssim = ssim_loss(rendered_image, gt_image)

        depth_ramp = min(1.0, step / (config.iterations * 0.2))
        loss_depth = depth_tv_loss(result.depth, gt_image)

        loss = (
            0.8 * loss_l1
            + 0.2 * loss_ssim
            + depth_ramp * config.depth_reg_weight * loss_depth
        )

        losses = {
            "l1": loss_l1.item(),
            "depth": loss_depth.item(),
        }

        # Floater reduction regularization
        if config.opacity_reg_weight > 0:
            activated = torch.sigmoid(self.params["opacities"].squeeze(-1))
            loss = loss + config.opacity_reg_weight * opacity_reg_loss(activated)
        if config.scale_reg_weight > 0:
            loss = loss + config.scale_reg_weight * scale_reg_loss(
                self.params["scales"]
            )
        if config.flatten_reg_weight > 0:
            loss = loss + config.flatten_reg_weight * flatten_reg_loss(
                self.params["scales"]
            )

        # 2DGS-specific losses
        if config.mode == "2dgs":
            if (
                config.distortion_weight > 0
                and result.distortion is not None
            ):
                loss = loss + config.distortion_weight * distortion_loss(
                    result.distortion
                )
            if (
                config.normal_weight > 0
                and result.normals is not None
                and result.surf_normals is not None
            ):
                loss = loss + config.normal_weight * normal_consistency_loss(
                    result.normals, result.surf_normals
                )

        # Appearance identity regularization — keep transform close to identity
        if config.appearance_embeddings and self.appearance_embeds is not None:
            embed = self.appearance_embeds(
                torch.tensor(idx, device=self.device)
            )
            affine = self.appearance_mlp(embed)
            identity_target = torch.cat([
                torch.eye(3, device=self.device).flatten(),
                torch.zeros(3, device=self.device),
            ])
            loss = loss + config.appearance_reg_weight * torch.nn.functional.mse_loss(
                affine, identity_target
            )

        losses["total"] = loss.item()

        # Backward
        loss.backward()

        with torch.no_grad():
            # Update learning rate for positions (exponential decay with warmup)
            lr_init = 1.6e-4
            lr_final = 1.6e-6
            t = step / config.iterations
            lr = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)

            # Position LR warmup: start at 1% and ramp up (matches reference position_lr_delay_mult=0.01)
            lr_delay_mult = 0.01
            warmup_steps = max(1, int(config.iterations * 0.01))
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(step / warmup_steps, 1.0)
            )
            lr = lr * delay_rate

            self.optimizers["means"].param_groups[0]["lr"] = lr

            # Step optimizers
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            # Normalize quaternions
            self.params["quats"].data = F.normalize(
                self.params["quats"].data, dim=-1
            )

            # Densification strategy step
            refine_stop = int(
                config.iterations * config.densify_until_pct
            )
            if step < refine_stop:
                self.strategy.step_post_backward(
                    params=self.params,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=result.info,
                    packed=False,
                )

            # TIDI-GS pruning
            if self.pruner is not None:
                self.pruner.resize(len(self.params["means"]))
                self.pruner.update(result.info, step)

                if (
                    step > 1000
                    and step % 1000 == 0
                    and step < refine_stop
                ):
                    prune_mask = self.pruner.get_prune_mask(self.params)
                    if prune_mask.any():
                        n_pruned = prune_mask.sum().item()
                        self._prune_gaussians(prune_mask)
                        logger.info(
                            f"TIDI pruned {n_pruned} floaters at step {step}"
                        )

        return losses

    def _prune_gaussians(self, mask: torch.Tensor):
        """Remove Gaussians indicated by boolean mask from params + optimizers."""
        keep = ~mask
        for key in self.params:
            self.params[key] = torch.nn.Parameter(
                self.params[key].data[keep]
            )

        # Preserve appearance optimizers across rebuild
        saved_appearance_opts = {}
        for k in ("appearance_embeds", "appearance_mlp"):
            if k in self.optimizers:
                saved_appearance_opts[k] = self.optimizers[k]

        # Rebuild Gaussian param optimizers
        self._init_optimizers()

        # Restore appearance optimizers
        self.optimizers.update(saved_appearance_opts)

        # Resize strategy state accumulators (grad2d, count)
        if self.strategy_state is not None:
            for key in ("grad2d", "count"):
                if key in self.strategy_state and self.strategy_state[key] is not None:
                    self.strategy_state[key] = self.strategy_state[key][keep]

        # Resize pruner
        if self.pruner is not None:
            self.pruner.resize(len(self.params["means"]))

    def train(self, callbacks: TrainerCallbacks, output_path: Path | None = None) -> int:
        """Main training loop. Returns final Gaussian count."""
        if output_path is not None:
            self._output_path = output_path
        config = self.config
        num_views = len(self.cameras_data)

        # Set up preview camera (first training view, reduced resolution)
        preview_cam = self.cameras_data[0]
        preview_scale = max(1, preview_cam["width"] // 640)
        preview_W = preview_cam["width"] // preview_scale
        preview_H = preview_cam["height"] // preview_scale
        preview_K = preview_cam["K"].copy()
        preview_K[:2] /= preview_scale
        preview_viewmat = torch.tensor(
            preview_cam["viewmat"], device=self.device
        ).unsqueeze(0)
        preview_K_t = torch.tensor(preview_K, device=self.device).unsqueeze(0)

        logger.info(
            f"Starting training: {config.iterations} iterations, "
            f"{num_views} views, sh_degree={config.sh_degree}, "
            f"mode={config.mode}"
        )

        start_step = self.current_step
        for step in range(start_step, config.iterations):
            self.current_step = step
            losses = self.train_step(step)

            # Progress logging
            if step % 1000 == 0 or step == config.iterations - 1:
                n_gaussians = len(self.params["means"])
                logger.info(
                    f"  Step {step}/{config.iterations}: "
                    f"loss={losses['total']:.4f}, "
                    f"L1={losses['l1']:.4f}, "
                    f"depth={losses['depth']:.4f}, "
                    f"N={n_gaussians}"
                )
                if callbacks.progress:
                    callbacks.progress(step / config.iterations)

            # Preview rendering
            if (
                callbacks.preview
                and step % callbacks.preview_every == 0
            ):
                jpeg_bytes = self.rasterizer.rasterize_preview(
                    self.params,
                    preview_viewmat,
                    preview_K_t,
                    preview_W,
                    preview_H,
                    config.sh_degree,
                )
                n_gs = len(self.params["means"])
                callbacks.preview(
                    step, config.iterations, losses["total"], n_gs, jpeg_bytes
                )

            # 3D snapshot export
            if (
                callbacks.snapshot
                and step > 0
                and step % callbacks.snapshot_every == 0
            ):
                self._export_snapshot(step, callbacks, losses["total"])

            # Auto-save checkpoint every 5000 steps
            if (
                output_path is not None
                and step > 0
                and step % 5000 == 0
            ):
                ckpt_path = output_path.parent / f"checkpoint_{step}.pt"
                self.save_checkpoint(ckpt_path)
                # Keep only latest 2 checkpoints
                ckpts = sorted(
                    output_path.parent.glob("checkpoint_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                for old in ckpts[:-2]:
                    old.unlink()

        self.current_step = config.iterations

        n_final = len(self.params["means"])
        logger.info(f"Training complete: {n_final} Gaussians")

        if callbacks.progress:
            callbacks.progress(1.0)

        return n_final

    def _export_snapshot(
        self, step: int, callbacks: TrainerCallbacks, loss: float
    ):
        """Export a 3D PLY snapshot during training."""
        with torch.no_grad():
            snapshot_name = f"snapshot_{step:06d}.ply"
            snapshot_path = self._output_path.parent / snapshot_name
            export_splats(
                means=self.params["means"].detach(),
                scales=self.params["scales"].detach(),
                quats=self.params["quats"].detach(),
                opacities=self.params["opacities"].detach().squeeze(-1),
                sh0=self.params["sh0"].detach(),
                shN=self.params["shN"].detach(),
                format="ply",
                save_to=str(snapshot_path),
            )
            # Delete previous snapshot to avoid disk bloat
            prev_step = step - callbacks.snapshot_every
            if prev_step > 0:
                prev_snapshot = (
                    self._output_path.parent
                    / f"snapshot_{prev_step:06d}.ply"
                )
                if prev_snapshot.exists():
                    prev_snapshot.unlink()

            n_gs = len(self.params["means"])
            callbacks.snapshot(
                step, self.config.iterations, loss, n_gs, snapshot_name
            )

    def export(self, output_path: Path):
        """Export trained Gaussians to PLY."""
        self._output_path = output_path
        logger.info(f"Exporting to {output_path}")
        export_splats(
            means=self.params["means"].detach(),
            scales=self.params["scales"].detach(),
            quats=self.params["quats"].detach(),
            opacities=self.params["opacities"].detach().squeeze(-1),
            sh0=self.params["sh0"].detach(),
            shN=self.params["shN"].detach(),
            format="ply",
            save_to=str(output_path),
        )

    def save_checkpoint(self, path: Path):
        """Save training checkpoint for resume."""
        state = {
            "step": self.current_step,
            "params": {k: v.data for k, v in self.params.items()},
            "optimizer_states": {
                k: v.state_dict()
                for k, v in self.optimizers.items()
                if k not in ("appearance_embeds", "appearance_mlp")
            },
            "strategy_state": self.strategy_state,
            "config": asdict(self.config),
        }
        if self.appearance_embeds is not None:
            state["appearance_embeds"] = self.appearance_embeds.state_dict()
            state["appearance_mlp"] = self.appearance_mlp.state_dict()
            state["optimizer_states"]["appearance_embeds"] = self.optimizers[
                "appearance_embeds"
            ].state_dict()
            state["optimizer_states"]["appearance_mlp"] = self.optimizers[
                "appearance_mlp"
            ].state_dict()
        torch.save(state, str(path))

    def load_checkpoint(self, path: Path):
        """Load training checkpoint to resume."""
        ckpt = torch.load(str(path), map_location="cuda")
        self.current_step = ckpt["step"]
        for k, v in ckpt["params"].items():
            self.params[k] = torch.nn.Parameter(v)

        # Rebuild base optimizers
        self._init_optimizers()

        # Restore appearance model weights + recreate their optimizers
        if self.appearance_embeds is not None and "appearance_embeds" in ckpt:
            self.appearance_embeds.load_state_dict(ckpt["appearance_embeds"])
            self.appearance_mlp.load_state_dict(ckpt["appearance_mlp"])
            self.optimizers["appearance_embeds"] = torch.optim.Adam(
                self.appearance_embeds.parameters(), lr=1e-4
            )
            self.optimizers["appearance_mlp"] = torch.optim.Adam(
                self.appearance_mlp.parameters(), lr=1e-4
            )

        # Load saved optimizer states
        for k, state in ckpt["optimizer_states"].items():
            if k in self.optimizers:
                self.optimizers[k].load_state_dict(state)

        self.strategy_state = ckpt["strategy_state"]

    def cleanup(self):
        """Free GPU memory."""
        del self.params, self.gt_images, self.optimizers, self.strategy_state
        if self.appearance_embeds is not None:
            del self.appearance_embeds, self.appearance_mlp
        torch.cuda.empty_cache()
