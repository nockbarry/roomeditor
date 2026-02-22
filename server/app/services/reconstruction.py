import asyncio
import base64
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.database import async_session
from app.models import Job, Project
from app.routes.ws import broadcast_job_progress, broadcast_training_preview

logger = logging.getLogger(__name__)

# Thread pool for CPU/GPU-bound pipeline steps
_executor = ThreadPoolExecutor(max_workers=1)


async def update_job_progress(
    job_id: str,
    status: str,
    progress: float,
    current_step: str | None = None,
    error_message: str | None = None,
):
    """Update job in DB and broadcast to WebSocket clients."""
    async with async_session() as db:
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = status
            job.progress = progress
            job.current_step = current_step
            job.error_message = error_message
            await db.commit()

    await broadcast_job_progress(job_id, {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "current_step": current_step,
        "error_message": error_message,
    })


async def run_reconstruction_pipeline(project_id: str, job_id: str, training_config: dict | None = None):
    """Run the full reconstruction pipeline in the background.

    Pipeline: extract frames -> select keyframes -> COLMAP SfM -> gsplat training -> export
    """
    project_dir = settings.data_dir / "projects" / project_id
    source_dir = project_dir / "sources"
    frames_dir = project_dir / "frames"
    colmap_dir = project_dir / "colmap"
    ply_path = project_dir / "scene.ply"

    loop = asyncio.get_event_loop()

    try:
        # ---- Step 1: Extract frames from all videos + photos ----
        await update_job_progress(job_id, "running", 0.0, "Extracting frames")

        from pipeline.extract_frames import extract_frames_multi

        async def _frame_progress(frac: float):
            overall = frac * 0.08
            await update_job_progress(
                job_id, "running", overall, "Extracting frames"
            )

        frame_count = await extract_frames_multi(
            source_dir=source_dir,
            output_dir=frames_dir,
            fps=settings.ffmpeg_fps,
            ffmpeg_path=settings.ffmpeg_path,
            progress_callback=_frame_progress,
        )
        logger.info(f"Extracted {frame_count} frames total")

        # ---- Step 1.5: Select keyframes (skip blurry/redundant) ----
        await update_job_progress(job_id, "running", 0.08, "Selecting keyframes")

        from pipeline.select_keyframes import select_keyframes

        kept, removed = await select_keyframes(frames_dir)
        logger.info(f"Keyframe selection: kept {kept}, removed {removed}")

        # ---- Step 2: Run SfM or feed-forward ----
        cfg = training_config or {}
        sfm_backend = cfg.get("sfm_backend", "colmap")

        # SPFSplat: pose-free feed-forward (alternative to AnySplat)
        if sfm_backend == "spfsplat":
            await update_job_progress(job_id, "running", 0.15, "Running SPFSplat (feed-forward)")

            from pipeline.run_spfsplat_subprocess import run_spfsplat_subprocess

            max_views = cfg.get("spfsplat_max_views", 8)

            def _spfsplat_progress(frac):
                overall = 0.15 + frac * 0.75
                asyncio.run_coroutine_threadsafe(
                    update_job_progress(job_id, "running", overall, "Running SPFSplat"),
                    loop,
                )

            def _run_spfsplat():
                return run_spfsplat_subprocess(
                    images_dir=frames_dir,
                    output_ply=ply_path,
                    max_views=max_views,
                    progress_callback=_spfsplat_progress,
                )

            n_gaussians = await loop.run_in_executor(_executor, _run_spfsplat)

            await update_job_progress(job_id, "running", 0.95, "Finalizing")

            for snapshot in project_dir.glob("snapshot_*.ply"):
                snapshot.unlink()

            async with async_session() as db:
                result = await db.execute(select(Project).where(Project.id == project_id))
                project = result.scalar_one_or_none()
                if project:
                    project.status = "ready"
                    project.gaussian_count = n_gaussians
                    await db.commit()

            await update_job_progress(job_id, "completed", 1.0, "Done")
            logger.info(f"SPFSplat complete for project {project_id}: {n_gaussians} Gaussians")
            return

        # AnySplat: skip SfM + training entirely
        if sfm_backend == "anysplat":
            await update_job_progress(job_id, "running", 0.15, "Running AnySplat (feed-forward)")

            from pipeline.run_anysplat_subprocess import run_anysplat_subprocess

            max_views = cfg.get("anysplat_max_views", 32)
            use_chunked = cfg.get("anysplat_chunked", False)

            def _anysplat_progress(frac):
                overall = 0.15 + frac * 0.75
                asyncio.run_coroutine_threadsafe(
                    update_job_progress(job_id, "running", overall, "Running AnySplat"),
                    loop,
                )

            def _run_anysplat():
                return run_anysplat_subprocess(
                    images_dir=frames_dir,
                    output_ply=ply_path,
                    max_views=max_views,
                    chunked=use_chunked,
                    progress_callback=_anysplat_progress,
                )

            n_gaussians = await loop.run_in_executor(_executor, _run_anysplat)

            # Deduplicate Gaussians if chunked mode was used
            if use_chunked:
                await update_job_progress(job_id, "running", 0.92, "Deduplicating chunk overlaps")
                from pipeline.compress_splat import deduplicate_gaussians
                dedup_stats = await deduplicate_gaussians(ply_path)
                n_gaussians = dedup_stats["n_after"]
                logger.info(f"Chunked dedup: {dedup_stats}")

            await update_job_progress(job_id, "running", 0.95, "Finalizing")

            # Clean up snapshot files
            for snapshot in project_dir.glob("snapshot_*.ply"):
                snapshot.unlink()

            async with async_session() as db:
                result = await db.execute(select(Project).where(Project.id == project_id))
                project = result.scalar_one_or_none()
                if project:
                    project.status = "ready"
                    project.gaussian_count = n_gaussians
                    await db.commit()

            await update_job_progress(job_id, "completed", 1.0, "Done")
            logger.info(f"AnySplat complete for project {project_id}: {n_gaussians} Gaussians")
            return

        def _make_sfm_progress_callback(step_name):
            """Bridge SfM progress from thread to async WebSocket."""
            def on_progress(frac):
                overall = 0.12 + frac * 0.28
                asyncio.run_coroutine_threadsafe(
                    update_job_progress(
                        job_id, "running", overall, step_name
                    ),
                    loop,
                )
            return on_progress

        if sfm_backend == "mast3r":
            await update_job_progress(job_id, "running", 0.12, "Running MASt3R (pose estimation)")

            from pipeline.run_mast3r import run_mast3r

            mast3r_image_size = cfg.get("mast3r_image_size", 512)

            def _run_mast3r():
                return run_mast3r(
                    images_dir=frames_dir,
                    output_dir=colmap_dir,
                    image_size=mast3r_image_size,
                    progress_callback=_make_sfm_progress_callback("Running MASt3R (pose estimation)"),
                )

            model_dir, colmap_meta = await loop.run_in_executor(_executor, _run_mast3r)
        else:
            await update_job_progress(job_id, "running", 0.12, "Running COLMAP (SfM)")

            from pipeline.run_colmap import run_colmap

            def _run_colmap():
                return run_colmap(
                    images_dir=frames_dir,
                    output_dir=colmap_dir,
                    progress_callback=_make_sfm_progress_callback("Running COLMAP (SfM)"),
                )

            model_dir, colmap_meta = await loop.run_in_executor(_executor, _run_colmap)

        if colmap_meta.get("num_reconstructions", 1) > 1:
            logger.warning(
                f"Multiple scenes detected in input ({colmap_meta['num_reconstructions']} reconstructions). "
                f"Using largest with {colmap_meta['registered_images']} images."
            )

        logger.info(f"SfM complete ({sfm_backend}): {colmap_meta}")

        await update_job_progress(job_id, "running", 0.40, "Training 3D Gaussians")

        # ---- Step 3: Train Gaussians (GPU-bound, run in thread) ----
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        def _make_preview_callback():
            """Create a callback that bridges the training thread to async WebSocket."""
            def on_preview(step, total_steps, loss, n_gaussians, jpeg_bytes):
                preview_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                data = {
                    "type": "training_preview",
                    "job_id": job_id,
                    "step": step,
                    "total_steps": total_steps,
                    "loss": round(loss, 5),
                    "n_gaussians": n_gaussians,
                    "preview_base64": preview_b64,
                }
                asyncio.run_coroutine_threadsafe(
                    broadcast_training_preview(job_id, data), loop
                )
            return on_preview

        def _make_progress_callback():
            """Create a callback that sends training progress updates."""
            def on_progress(frac):
                overall = 0.40 + frac * 0.55
                asyncio.run_coroutine_threadsafe(
                    update_job_progress(
                        job_id, "running", overall, "Training 3D Gaussians"
                    ),
                    loop,
                )
            return on_progress

        def _make_snapshot_callback():
            """Create a callback that notifies clients when a 3D snapshot is available."""
            def on_snapshot(step, total_steps, loss, n_gaussians, snapshot_filename):
                data = {
                    "type": "training_snapshot",
                    "job_id": job_id,
                    "step": step,
                    "total_steps": total_steps,
                    "loss": round(loss, 5),
                    "n_gaussians": n_gaussians,
                    "snapshot_url": f"/data/{project_id}/{snapshot_filename}",
                }
                asyncio.run_coroutine_threadsafe(
                    broadcast_training_preview(job_id, data), loop
                )
            return on_snapshot

        def _train():
            config = TrainerConfig(
                iterations=cfg.get("iterations", settings.gsplat_iterations),
                sh_degree=cfg.get("sh_degree", settings.gsplat_sh_degree),
                mode=cfg.get("mode", "3dgs"),
                depth_reg_weight=cfg.get("depth_reg_weight", 0.0),
                opacity_reg_weight=cfg.get("opacity_reg_weight", 0.0),
                scale_reg_weight=cfg.get("scale_reg_weight", 0.0),
                flatten_reg_weight=cfg.get("flatten_reg_weight", 0.0),
                distortion_weight=cfg.get("distortion_weight", 0.0),
                normal_weight=cfg.get("normal_weight", 0.0),
                prune_opa=cfg.get("prune_opa", 0.005),
                densify_until_pct=cfg.get("densify_until_pct", 0.5),
                appearance_embeddings=cfg.get("appearance_embeddings", False),
                tidi_pruning=cfg.get("tidi_pruning", False),
            )
            trainer = GaussianTrainer(config)
            points3d, colors3d = trainer.load_data(model_dir, frames_dir)
            trainer.init_params(points3d, colors3d)
            trainer.init_strategy()

            callbacks = TrainerCallbacks(
                progress=_make_progress_callback(),
                preview=_make_preview_callback(),
                snapshot=_make_snapshot_callback(),
            )

            # Resume from checkpoint if specified
            resume_from = cfg.get("resume_from")
            if resume_from:
                ckpt_path = project_dir / resume_from
                if ckpt_path.exists():
                    trainer.load_checkpoint(ckpt_path)
                    logger.info(f"Resumed from checkpoint: {resume_from}")

            n_gaussians = trainer.train(callbacks, output_path=ply_path)
            trainer.export(ply_path)
            trainer.cleanup()
            return n_gaussians

        gaussian_count = await loop.run_in_executor(_executor, _train)

        await update_job_progress(job_id, "running", 0.95, "Finalizing")

        # Clean up snapshot files
        for snapshot in project_dir.glob("snapshot_*.ply"):
            snapshot.unlink()

        # Update project status
        async with async_session() as db:
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            if project:
                project.status = "ready"
                project.gaussian_count = gaussian_count
                await db.commit()

        await update_job_progress(job_id, "completed", 1.0, "Done")
        logger.info(
            f"Reconstruction complete for project {project_id}: "
            f"{gaussian_count} Gaussians"
        )

    except Exception as e:
        logger.error(f"Reconstruction failed for project {project_id}: {e}")
        logger.error(traceback.format_exc())

        await update_job_progress(
            job_id, "failed", 0.0, error_message=str(e)
        )

        async with async_session() as db:
            result = await db.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()
            if project:
                project.status = "error"
                project.error_message = str(e)
                await db.commit()
