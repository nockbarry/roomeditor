"""Post-processing endpoints — prune, refine, quality stats."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Project

logger = logging.getLogger(__name__)
_refine_executor = ThreadPoolExecutor(max_workers=1)

router = APIRouter(prefix="/api/projects", tags=["postprocess"])


# --- Prune ---


class PruneRequest(BaseModel):
    opacity_threshold: float = 0.01
    max_gaussians: int = 2_000_000


class PruneResult(BaseModel):
    n_before: int
    n_after: int
    n_pruned: int


@router.post("/{project_id}/prune", response_model=PruneResult)
async def prune_splat(
    project_id: str,
    body: PruneRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Prune low-opacity gaussians from scene.ply."""
    if body is None:
        body = PruneRequest()

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ply_path = settings.data_dir / "projects" / project_id / "scene.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    from pipeline.compress_splat import prune_gaussians

    stats = await prune_gaussians(
        ply_path, ply_path,
        opacity_threshold=body.opacity_threshold,
        max_gaussians=body.max_gaussians,
    )

    # Update gaussian count in DB
    project.gaussian_count = stats["n_after"]
    await db.commit()

    return PruneResult(**stats)


# --- Quality Stats ---


class QualityStatsResult(BaseModel):
    n_gaussians: int = 0
    n_effective: int = 0
    frac_transparent: float = 0.0
    frac_opaque: float = 0.0
    opacity_mean: float = 0.0
    log_scale_mean: float = 0.0
    log_scale_std: float = 0.0
    frac_scale_outlier: float = 0.0
    mean_nn_dist: float = 0.0
    bbox_volume: float = 0.0
    density: float = 0.0


@router.get("/{project_id}/quality-stats", response_model=QualityStatsResult)
async def get_quality_stats(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Compute gaussian quality statistics from scene.ply."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ply_path = settings.data_dir / "projects" / project_id / "scene.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    from pipeline.evaluate_noreference import compute_gaussian_stats

    stats = compute_gaussian_stats(ply_path)

    return QualityStatsResult(
        n_gaussians=stats.n_gaussians,
        n_effective=stats.n_effective,
        frac_transparent=stats.frac_transparent,
        frac_opaque=stats.frac_opaque,
        opacity_mean=stats.opacity_mean,
        log_scale_mean=stats.log_scale_mean,
        log_scale_std=stats.log_scale_std,
        frac_scale_outlier=stats.frac_scale_outlier,
        mean_nn_dist=stats.mean_nn_dist,
        bbox_volume=stats.bbox_volume,
        density=stats.density,
    )


# --- Refine (Training from AnySplat output) ---


class RefineRequest(BaseModel):
    iterations: int = 2000
    mode: str = "3dgs"  # "3dgs" or "2dgs"


class RefineResult(BaseModel):
    status: str
    n_gaussians: int
    iterations_run: int
    ply_url: str


def _run_refine(
    project_dir: Path, frames_dir: Path, iterations: int, mode: str,
    on_snapshot=None, on_progress=None,
) -> dict:
    """Run training refinement in a thread (blocking, GPU-bound)."""
    import torch
    import numpy as np
    from plyfile import PlyData

    ply_path = project_dir / "scene.ply"
    cameras_path = project_dir / "cameras.json"

    if not cameras_path.exists():
        raise RuntimeError("No cameras.json found — rebuild with AnySplat first")

    # Load camera data from AnySplat's cameras.json
    with open(cameras_path) as f:
        cam_data = json.load(f)

    resolution = cam_data.get("resolution", 448)
    default_focal = cam_data.get("focal_length_px", resolution * 0.85)

    cameras_data = []
    for cam in cam_data["cameras"]:
        frame_path = frames_dir / cam["frame"]
        if not frame_path.exists():
            continue

        # Build viewmat from camera-to-world transform
        transform = np.array(cam["transform"], dtype=np.float32)
        if transform.shape == (3, 4):
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :] = transform
        else:
            c2w = transform

        # Invert camera-to-world to get world-to-camera (viewmat)
        viewmat = np.linalg.inv(c2w).astype(np.float32)

        # Build intrinsics
        fx = cam.get("fx", default_focal)
        fy = cam.get("fy", default_focal)
        cx = cam.get("cx", resolution / 2)
        cy = cam.get("cy", resolution / 2)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        cameras_data.append({
            "image_path": str(frame_path),
            "K": K,
            "viewmat": viewmat,
            "width": resolution,
            "height": resolution,
        })

    if len(cameras_data) < 2:
        raise RuntimeError(f"Only {len(cameras_data)} valid camera views found, need at least 2")

    # Load existing PLY as initialization
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n_pts = len(vertex.data)

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Extract DC color from SH
    if "f_dc_0" in vertex.data.dtype.names:
        C0 = 0.28209479177387814
        r = np.array(vertex["f_dc_0"]) * C0 + 0.5
        g = np.array(vertex["f_dc_1"]) * C0 + 0.5
        b = np.array(vertex["f_dc_2"]) * C0 + 0.5
        colors = np.stack([r, g, b], axis=-1).clip(0, 1).astype(np.float32)
    else:
        colors = np.ones((n_pts, 3), dtype=np.float32) * 0.5

    logger.info(f"Refining {n_pts:,} gaussians with {len(cameras_data)} views for {iterations} iterations")

    # Use the trainer
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig, TrainerCallbacks

    config = TrainerConfig(
        iterations=iterations,
        sh_degree=0,  # Start with DC only for fast refinement
        mode=mode,
        depth_reg_weight=0.05,
        opacity_reg_weight=0.01,
        scale_reg_weight=0.01,
        prune_opa=0.005,
        densify_until_pct=0.5,
    )

    trainer = GaussianTrainer(config)
    trainer.cameras_data = cameras_data

    # Load images
    from PIL import Image as PILImage
    trainer.gt_images = []
    for cam in cameras_data:
        img = PILImage.open(cam["image_path"]).convert("RGB")
        if img.size != (cam["width"], cam["height"]):
            img = img.resize((cam["width"], cam["height"]), PILImage.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        trainer.gt_images.append(
            torch.tensor(arr, dtype=torch.float32, device="cuda")
        )

    # Initialize from existing PLY points
    trainer.init_params(means, colors)
    trainer.init_strategy()

    output_path = project_dir / "refined.ply"
    callbacks = TrainerCallbacks(
        snapshot=on_snapshot,
        progress=on_progress,
        snapshot_every=500,
    )
    n_final = trainer.train(callbacks, output_path)

    # Export and replace scene.ply
    trainer.export(output_path)

    import shutil
    # Backup original
    backup = project_dir / "scene_pre_refine.ply"
    if not backup.exists():
        shutil.copy2(str(ply_path), str(backup))
    shutil.copy2(str(output_path), str(ply_path))

    trainer.cleanup()

    return {
        "n_gaussians": n_final,
        "iterations_run": iterations,
    }


@router.post("/{project_id}/refine", response_model=RefineResult)
async def refine_splat(
    project_id: str,
    body: RefineRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Refine AnySplat output with short training pass."""
    if body is None:
        body = RefineRequest()

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = settings.data_dir / "projects" / project_id
    frames_dir = project_dir / "frames"
    ply_path = project_dir / "scene.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found — rebuild first")

    import asyncio
    from app.routes.ws import broadcast_stream_update

    loop = asyncio.get_event_loop()

    def on_snapshot(step, total, loss, n_gs, snapshot_filename):
        asyncio.run_coroutine_threadsafe(
            broadcast_stream_update(project_id, {
                "type": "refine_snapshot",
                "step": step,
                "total_steps": total,
                "loss": round(loss, 5),
                "n_gaussians": n_gs,
                "snapshot_url": f"/data/{project_id}/{snapshot_filename}",
            }),
            loop,
        )

    def on_progress(frac):
        asyncio.run_coroutine_threadsafe(
            broadcast_stream_update(project_id, {
                "type": "refine_progress",
                "progress": round(frac, 3),
            }),
            loop,
        )

    try:
        result_data = await loop.run_in_executor(
            _refine_executor,
            lambda: _run_refine(
                project_dir, frames_dir, body.iterations, body.mode,
                on_snapshot=on_snapshot, on_progress=on_progress,
            ),
        )
    except Exception as e:
        logger.error(f"Refine failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    project.gaussian_count = result_data["n_gaussians"]
    await db.commit()

    return RefineResult(
        status="refined",
        n_gaussians=result_data["n_gaussians"],
        iterations_run=result_data["iterations_run"],
        ply_url=f"/data/{project_id}/scene.ply",
    )
