"""Post-processing endpoints — prune, refine, quality stats."""

import base64
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

# Track active trainers for stop support
_active_trainers: dict[str, object] = {}

# --- Refinement Presets ---

REFINE_PRESETS = {
    "conservative": {
        "label": "Conservative",
        "description": "Appearance-only refinement. Positions frozen, no densification. Safest option.",
        "config": {
            "iterations": 1000,
            "sh_degree": 1,
            "means_lr": 1.6e-4,
            "means_lr_final": 1.6e-6,
            "scales_lr": 5e-4,
            "quats_lr": 1e-4,
            "opacities_lr": 5e-3,
            "sh0_lr": 1e-3,
            "shN_lr": 1e-3,
            "freeze_positions_pct": 1.0,
            "densify_enabled": False,
            "reset_opacity": False,
            "depth_reg_weight": 0.05,
            "opacity_reg_weight": 0.01,
            "scale_reg_weight": 0.01,
        },
    },
    "balanced": {
        "label": "Balanced",
        "description": "Moderate refinement with conservative densification. Recommended default.",
        "config": {
            "iterations": 2000,
            "sh_degree": 1,
            "means_lr": 1.6e-5,
            "means_lr_final": 1.6e-7,
            "scales_lr": 1e-3,
            "quats_lr": 2e-4,
            "opacities_lr": 1e-2,
            "sh0_lr": 5e-4,
            "shN_lr": 5e-4,
            "freeze_positions_pct": 0.1,
            "densify_enabled": True,
            "reset_opacity": False,
            "grow_grad2d": 0.0005,
            "depth_reg_weight": 0.05,
            "opacity_reg_weight": 0.01,
            "scale_reg_weight": 0.01,
        },
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "Full retrain with higher learning rates and full densification.",
        "config": {
            "iterations": 5000,
            "sh_degree": 2,
            "means_lr": 8e-5,
            "means_lr_final": 8e-7,
            "scales_lr": 3e-3,
            "quats_lr": 5e-4,
            "opacities_lr": 3e-2,
            "sh0_lr": 1.5e-3,
            "shN_lr": 1.5e-3,
            "freeze_positions_pct": 0.05,
            "densify_enabled": True,
            "reset_opacity": False,
            "grow_grad2d": 0.0003,
            "depth_reg_weight": 0.05,
            "opacity_reg_weight": 0.01,
            "scale_reg_weight": 0.01,
        },
    },
}

router = APIRouter(prefix="/api/projects", tags=["postprocess"])


# --- Model Info ---


def _ply_gaussian_count(path: Path) -> int | None:
    """Read gaussian count from a PLY header without loading the whole file."""
    try:
        with open(path, "rb") as f:
            header = f.read(4096).decode("ascii", errors="ignore")
        import re
        m = re.search(r"element vertex (\d+)", header)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _file_info(project_id: str, project_dir: Path, filename: str) -> dict | None:
    """Return file info dict if the file exists, else None."""
    path = project_dir / filename
    if not path.exists():
        return None
    size_bytes = path.stat().st_size
    info: dict = {
        "filename": filename,
        "url": f"/data/{project_id}/{filename}",
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1048576),
    }
    if filename.endswith(".ply"):
        gc = _ply_gaussian_count(path)
        if gc is not None:
            info["gaussian_count"] = gc
    return info


@router.get("/{project_id}/model-info")
async def get_model_info(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return pipeline stage provenance and available formats for a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = settings.data_dir / "projects" / project_id

    has_anysplat = (project_dir / "anysplat_output.ply").exists()
    has_scene = (project_dir / "scene.ply").exists()
    has_pre_prune = (project_dir / "scene_pre_prune.ply").exists()
    has_pre_refine = (project_dir / "scene_pre_refine.ply").exists()

    stages = []

    # Stage 1: AnySplat — raw output before any postprocessing
    if has_anysplat:
        files = {}
        info = _file_info(project_id, project_dir, "anysplat_output.ply")
        if info:
            files["ply"] = info
        stages.append({
            "id": "anysplat",
            "label": "AnySplat",
            "description": "Feed-forward reconstruction from video frames",
            "exists": True,
            "files": files,
        })
    else:
        stages.append({
            "id": "anysplat",
            "label": "AnySplat",
            "description": "Feed-forward reconstruction from video frames",
            "exists": False,
            "files": {},
        })

    # Stage 2: Cleanup — auto-cleanup (opacity/scale/floater removal)
    # Cleanup output is the earliest pre-* backup or scene.ply itself
    cleanup_exists = has_scene and not has_pre_prune and not has_pre_refine and not has_anysplat
    # Actually, cleanup exists if scene.ply exists AND anysplat_output.ply also exists
    # (meaning scene.ply is the result of cleanup applied to anysplat_output.ply)
    # OR if there's a pre_prune/pre_refine backup (meaning cleanup happened before those)
    cleanup_exists = has_scene and has_anysplat
    if cleanup_exists:
        # The cleanup output file is:
        # - scene_pre_prune.ply if pruning happened after cleanup
        # - scene_pre_refine.ply if refine happened after cleanup (but no prune)
        # - scene.ply if nothing else happened after cleanup
        if has_pre_prune:
            cleanup_file = "scene_pre_prune.ply"
        elif has_pre_refine:
            cleanup_file = "scene_pre_refine.ply"
        else:
            cleanup_file = "scene.ply"

        files = {}
        info = _file_info(project_id, project_dir, cleanup_file)
        if info:
            files["ply"] = info
        stages.append({
            "id": "cleanup",
            "label": "Cleanup",
            "description": "Opacity, scale, and floater removal",
            "exists": True,
            "files": files,
        })
    else:
        stages.append({
            "id": "cleanup",
            "label": "Cleanup",
            "description": "Opacity, scale, and floater removal",
            "exists": False,
            "files": {},
        })

    # Stage 3: Prune — gaussian pruning
    if has_pre_prune:
        # Prune output is scene_pre_refine.ply if refine happened, else scene.ply
        if has_pre_refine:
            prune_file = "scene_pre_refine.ply"
        else:
            prune_file = "scene.ply"

        files = {}
        info = _file_info(project_id, project_dir, prune_file)
        if info:
            files["ply"] = info
        stages.append({
            "id": "prune",
            "label": "Prune",
            "description": "Low-opacity gaussian pruning",
            "exists": True,
            "files": files,
        })
    else:
        stages.append({
            "id": "prune",
            "label": "Prune",
            "description": "Low-opacity gaussian pruning",
            "exists": False,
            "files": {},
        })

    # Stage 4: Refine — training refinement
    if has_pre_refine:
        files = {}
        info = _file_info(project_id, project_dir, "scene.ply")
        if info:
            files["ply"] = info
        # SPZ is always generated from scene.ply
        spz_info = _file_info(project_id, project_dir, "scene.spz")
        if spz_info:
            files["spz"] = spz_info
        stages.append({
            "id": "refine",
            "label": "Refine",
            "description": "Training-based refinement",
            "exists": True,
            "files": files,
        })
    else:
        stages.append({
            "id": "refine",
            "label": "Refine",
            "description": "Training-based refinement",
            "exists": False,
            "files": {},
        })

    # Determine current (latest) stage
    if has_pre_refine:
        current_stage = "refine"
    elif has_pre_prune:
        current_stage = "prune"
    elif cleanup_exists:
        current_stage = "cleanup"
    elif has_anysplat:
        current_stage = "anysplat"
    else:
        current_stage = None

    # Current scene formats (scene.ply + scene.spz)
    formats = {}
    ply_info = _file_info(project_id, project_dir, "scene.ply")
    if ply_info:
        formats["ply"] = ply_info
    spz_info = _file_info(project_id, project_dir, "scene.spz")
    if spz_info:
        formats["spz"] = spz_info

    return {
        "stages": stages,
        "current_stage": current_stage,
        "formats": formats,
    }


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

    # Save before-prune backup for comparison
    import shutil
    backup_path = ply_path.parent / "scene_pre_prune.ply"
    if not backup_path.exists():
        shutil.copy2(str(ply_path), str(backup_path))

    from pipeline.compress_splat import prune_gaussians

    stats = await prune_gaussians(
        ply_path, ply_path,
        opacity_threshold=body.opacity_threshold,
        max_gaussians=body.max_gaussians,
    )

    # Regenerate SPZ + positions sidecar
    from utils.spz_convert import generate_spz_bundle
    generate_spz_bundle(ply_path.parent)

    # Update gaussian count in DB
    project.gaussian_count = stats["n_after"]
    await db.commit()

    return PruneResult(**stats)


# --- Comparison Info ---


@router.get("/{project_id}/comparison-info")
async def get_comparison_info(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return available before/after PLY pairs for comparison."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = settings.data_dir / "projects" / project_id
    pairs = []

    pre_prune = project_dir / "scene_pre_prune.ply"
    if pre_prune.exists():
        pairs.append({
            "label": "Pre-Prune",
            "before_url": f"/data/{project_id}/scene_pre_prune.ply",
            "after_url": f"/data/{project_id}/scene.ply",
        })

    pre_refine = project_dir / "scene_pre_refine.ply"
    if pre_refine.exists():
        pairs.append({
            "label": "Pre-Refine",
            "before_url": f"/data/{project_id}/scene_pre_refine.ply",
            "after_url": f"/data/{project_id}/scene.ply",
        })

    return {"pairs": pairs}


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
    preset: str = "balanced"  # "conservative", "balanced", or "aggressive"
    mode: str = "3dgs"  # "3dgs" or "2dgs"


class RefineResult(BaseModel):
    status: str
    n_gaussians: int
    iterations_run: int
    ply_url: str
    pre_stats: dict | None = None
    post_stats: dict | None = None


@router.get("/config/refine-presets")
async def get_refine_presets():
    """Return available refinement preset definitions."""
    return {
        name: {"label": p["label"], "description": p["description"], "iterations": p["config"]["iterations"]}
        for name, p in REFINE_PRESETS.items()
    }


def _split_train_test(n_views: int, test_every: int = 4) -> tuple[list[int], list[int]]:
    """Split view indices into train and test sets."""
    train = []
    test = []
    for i in range(n_views):
        if i % test_every == 0 and n_views > test_every:
            test.append(i)
        else:
            train.append(i)
    # Ensure at least 2 train views
    if len(train) < 2:
        train = list(range(n_views))
        test = []
    return train, test


def _run_refine(
    project_id: str,
    project_dir: Path, frames_dir: Path, preset: str = "balanced",
    mode: str = "3dgs",
    on_snapshot=None, on_progress=None, on_metrics=None, on_evaluation=None,
) -> dict:
    """Run training refinement in a thread (blocking, GPU-bound)."""
    import torch
    import numpy as np
    from plyfile import PlyData

    ply_path = project_dir / "scene.ply"
    cameras_path = project_dir / "cameras.json"

    if not cameras_path.exists():
        raise RuntimeError("No cameras.json found — rebuild with AnySplat first")

    # Resolve preset config
    preset_def = REFINE_PRESETS.get(preset, REFINE_PRESETS["balanced"])
    preset_config = preset_def["config"]

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

    # Load existing PLY — extract ALL parameters in their native space
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n_pts = len(vertex.data)
    field_names = vertex.data.dtype.names

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Extract scales (log-space), rotations, opacities (logit-space), SH0
    has_full_params = all(
        f in field_names for f in ("scale_0", "rot_0", "opacity", "f_dc_0")
    )

    if has_full_params:
        scales = np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
        ).astype(np.float32)
        quats = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=-1,
        ).astype(np.float32)
        opacities = np.array(vertex["opacity"]).astype(np.float32)
        sh0 = np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
        ).astype(np.float32)
    else:
        # Fallback: only positions and colors available
        scales = None
        quats = None
        opacities = None
        sh0 = None

    from pipeline.evaluate_noreference import compute_gaussian_stats

    pre_stats_obj = compute_gaussian_stats(ply_path)
    pre_stats = vars(pre_stats_obj) if hasattr(pre_stats_obj, '__dict__') else dict(pre_stats_obj)

    iterations = preset_config["iterations"]
    logger.info(f"Refining {n_pts:,} gaussians with {len(cameras_data)} views for {iterations} iterations (preset={preset})")

    # Use the trainer
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig, TrainerCallbacks

    config = TrainerConfig(
        iterations=iterations,
        sh_degree=preset_config.get("sh_degree", 1),
        mode=mode,
        depth_reg_weight=preset_config.get("depth_reg_weight", 0.05),
        opacity_reg_weight=preset_config.get("opacity_reg_weight", 0.01),
        scale_reg_weight=preset_config.get("scale_reg_weight", 0.01),
        prune_opa=0.005,
        densify_until_pct=0.5,
        means_lr=preset_config.get("means_lr", 1.6e-4),
        means_lr_final=preset_config.get("means_lr_final", 1.6e-6),
        scales_lr=preset_config.get("scales_lr", 5e-3),
        quats_lr=preset_config.get("quats_lr", 1e-3),
        opacities_lr=preset_config.get("opacities_lr", 5e-2),
        sh0_lr=preset_config.get("sh0_lr", 2.5e-3),
        shN_lr=preset_config.get("shN_lr", 2.5e-3),
        densify_enabled=preset_config.get("densify_enabled", True),
        reset_opacity=preset_config.get("reset_opacity", True),
        freeze_positions_pct=preset_config.get("freeze_positions_pct", 0.0),
        grow_grad2d=preset_config.get("grow_grad2d", 0.0002),
    )

    trainer = GaussianTrainer(config)

    # Store all cameras/images for evaluation before splitting
    all_cameras = list(cameras_data)

    # Load all images
    from PIL import Image as PILImage
    all_images = []
    for cam in all_cameras:
        img = PILImage.open(cam["image_path"]).convert("RGB")
        if img.size != (cam["width"], cam["height"]):
            img = img.resize((cam["width"], cam["height"]), PILImage.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        all_images.append(
            torch.tensor(arr, dtype=torch.float32, device="cuda")
        )

    # Train/test split
    train_indices, test_indices = _split_train_test(len(all_cameras), test_every=4)
    logger.info(f"Train/test split: {len(train_indices)} train, {len(test_indices)} test views")

    # Set up trainer with all data first
    trainer.cameras_data = all_cameras
    trainer.gt_images = all_images

    # Store references for evaluation during training
    trainer._all_cameras = all_cameras
    trainer._all_images = all_images
    trainer._eval_views = test_indices

    # Compute baseline metrics before training
    baseline = None
    if has_full_params:
        # Initialize from existing PLY with full parameters
        trainer.init_params_from_existing(means, scales, quats, opacities, sh0)
    else:
        # Fallback: extract colors and use sparse init
        C0 = 0.28209479177387814
        if "f_dc_0" in field_names:
            r = np.array(vertex["f_dc_0"]) * C0 + 0.5
            g = np.array(vertex["f_dc_1"]) * C0 + 0.5
            b = np.array(vertex["f_dc_2"]) * C0 + 0.5
            colors = np.stack([r, g, b], axis=-1).clip(0, 1).astype(np.float32)
        else:
            colors = np.ones((n_pts, 3), dtype=np.float32) * 0.5
        trainer.init_params(means, colors)

    trainer.init_strategy()

    # Compute baseline evaluation
    if test_indices:
        baseline = trainer.evaluate_views(test_indices, all_cameras, all_images)
        logger.info(f"Baseline: PSNR={baseline['mean_psnr']:.2f} dB, SSIM={baseline['mean_ssim']:.4f}")

    # Restrict training to train views only
    trainer.set_training_views(train_indices)

    # Register active trainer for stop support
    _active_trainers[project_id] = trainer

    output_path = project_dir / "refined.ply"
    callbacks = TrainerCallbacks(
        snapshot=on_snapshot,
        progress=on_progress,
        metrics=on_metrics,
        evaluation=on_evaluation,
        snapshot_every=500,
        metrics_every=50,
        eval_every=500,
    )

    try:
        n_final = trainer.train(callbacks, output_path)

        # Export and replace scene.ply
        trainer.export(output_path)

        import shutil
        # Backup original
        backup = project_dir / "scene_pre_refine.ply"
        if not backup.exists():
            shutil.copy2(str(ply_path), str(backup))
        shutil.copy2(str(output_path), str(ply_path))

        # Regenerate SPZ + positions sidecar
        from utils.spz_convert import generate_spz_bundle
        generate_spz_bundle(project_dir)
    finally:
        _active_trainers.pop(project_id, None)
        trainer.cleanup()

    post_stats_obj = compute_gaussian_stats(ply_path)
    post_stats = vars(post_stats_obj) if hasattr(post_stats_obj, '__dict__') else dict(post_stats_obj)

    return {
        "n_gaussians": n_final,
        "iterations_run": iterations,
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "baseline": baseline,
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

    if project_id in _active_trainers:
        raise HTTPException(status_code=409, detail="Refinement already in progress")

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

    def on_metrics(metrics_dict):
        asyncio.run_coroutine_threadsafe(
            broadcast_stream_update(project_id, {
                "type": "refine_metrics",
                **metrics_dict,
            }),
            loop,
        )

    def on_evaluation(eval_dict):
        asyncio.run_coroutine_threadsafe(
            broadcast_stream_update(project_id, {
                "type": "refine_eval",
                **eval_dict,
            }),
            loop,
        )

    try:
        result_data = await loop.run_in_executor(
            _refine_executor,
            lambda: _run_refine(
                project_id,
                project_dir, frames_dir,
                preset=body.preset, mode=body.mode,
                on_snapshot=on_snapshot, on_progress=on_progress,
                on_metrics=on_metrics, on_evaluation=on_evaluation,
            ),
        )
    except Exception as e:
        _active_trainers.pop(project_id, None)
        logger.error(f"Refine failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    project.gaussian_count = result_data["n_gaussians"]
    await db.commit()

    # Send baseline info in the final result
    baseline = result_data.get("baseline")

    return RefineResult(
        status="refined",
        n_gaussians=result_data["n_gaussians"],
        iterations_run=result_data["iterations_run"],
        ply_url=f"/data/{project_id}/scene.ply",
        pre_stats=result_data.get("pre_stats"),
        post_stats=result_data.get("post_stats"),
    )


@router.post("/{project_id}/refine/stop")
async def stop_refinement(
    project_id: str,
):
    """Request the active refinement to stop early."""
    trainer = _active_trainers.get(project_id)
    if trainer is None:
        raise HTTPException(status_code=404, detail="No active refinement for this project")
    trainer.request_stop()
    return {"status": "stop_requested"}


# --- Geometric Quality ---

_geo_executor = ThreadPoolExecutor(max_workers=1)


@router.get("/{project_id}/geometric-quality")
async def get_geometric_quality(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Compute geometric quality measures from scene.ply + cameras.json.

    Returns health_score (0-100), depth_consistency, edge_alignment,
    alpha_coverage, health_components, and gaussian_stats.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = settings.data_dir / "projects" / project_id
    ply_path = project_dir / "scene.ply"
    cameras_path = project_dir / "cameras.json"

    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")
    if not cameras_path.exists():
        raise HTTPException(status_code=400, detail="No cameras.json found")

    import asyncio
    from pipeline.geometric_quality import compute_geometric_quality

    loop = asyncio.get_event_loop()
    try:
        result_data = await loop.run_in_executor(
            _geo_executor,
            lambda: compute_geometric_quality(ply_path, cameras_path),
        )
    except Exception as e:
        logger.error(f"Geometric quality failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return result_data
