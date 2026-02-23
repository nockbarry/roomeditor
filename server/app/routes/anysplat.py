"""AnySplat Studio endpoints — interactive frame management and reconstruction."""

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Project
from app.schemas import AnySplatRunRequest, AnySplatRunResult, ExtractFramesRequest, FrameManifest

logger = logging.getLogger(__name__)
_anysplat_executor = ThreadPoolExecutor(max_workers=1)

router = APIRouter(prefix="/api/projects", tags=["anysplat"])


@router.post("/{project_id}/extract-frames", response_model=FrameManifest)
async def extract_frames(
    project_id: str,
    body: ExtractFramesRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Extract frames from uploaded sources and score them."""
    if body is None:
        body = ExtractFramesRequest()

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    source_dir = settings.data_dir / "projects" / project_id / "sources"
    frames_dir = settings.data_dir / "projects" / project_id / "frames"

    if not source_dir.exists() or not any(source_dir.iterdir()):
        raise HTTPException(status_code=400, detail="No files uploaded yet")

    from pipeline.extract_frames import extract_frames_multi

    _total, source_map = await extract_frames_multi(source_dir, frames_dir, fps=body.fps)

    from pipeline.select_keyframes import score_and_select_frames

    manifest = await score_and_select_frames(frames_dir, source_map=source_map)

    # Set reconstruction mode so frontend knows to use studio
    project.reconstruction_mode = "anysplat"
    if project.status == "created":
        project.status = "uploaded"
    await db.commit()

    return manifest


@router.get("/{project_id}/frames", response_model=FrameManifest)
async def get_frames(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List extracted frames from manifest or scan directory."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    frames_dir = settings.data_dir / "projects" / project_id / "frames"
    manifest_path = frames_dir / "frame_manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return manifest

    # No manifest — scan directory and build one
    if not frames_dir.exists():
        return {"frames": [], "total": 0, "selected_count": 0}

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    frame_infos = []
    for f in frames:
        frame_infos.append({
            "filename": f.name,
            "source_file": "",
            "source_type": "",
            "sharpness": 0.0,
            "selected": True,
        })

    manifest = {
        "frames": frame_infos,
        "total": len(frames),
        "selected_count": len(frames),
    }
    return manifest


class FrameUpdateRequest(BaseModel):
    updates: dict[str, bool]


@router.put("/{project_id}/frames", response_model=FrameManifest)
async def update_frame_selection(
    project_id: str,
    body: FrameUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Toggle frame selection flags in the manifest."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    frames_dir = settings.data_dir / "projects" / project_id / "frames"
    manifest_path = frames_dir / "frame_manifest.json"

    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="No frame manifest found. Extract frames first.")

    manifest = json.loads(manifest_path.read_text())

    for frame in manifest["frames"]:
        if frame["filename"] in body.updates:
            frame["selected"] = body.updates[frame["filename"]]

    manifest["selected_count"] = sum(1 for f in manifest["frames"] if f["selected"])
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


@router.post("/{project_id}/anysplat-run", response_model=AnySplatRunResult)
async def run_anysplat(
    project_id: str,
    body: AnySplatRunRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Run AnySplat on selected frames (synchronous, 3-10s)."""
    if body is None:
        body = AnySplatRunRequest()

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = settings.data_dir / "projects" / project_id
    frames_dir = project_dir / "frames"
    manifest_path = frames_dir / "frame_manifest.json"

    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="No frames extracted yet. Call extract-frames first.")

    manifest = json.loads(manifest_path.read_text())
    selected_frames = [f for f in manifest["frames"] if f["selected"]]

    if len(selected_frames) == 0:
        raise HTTPException(status_code=400, detail="No frames selected")

    # Create temp dir with symlinks to only selected frames (renumbered sequentially)
    temp_dir = project_dir / "_anysplat_input"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    for idx, frame_info in enumerate(selected_frames):
        src = frames_dir / frame_info["filename"]
        dst = temp_dir / f"frame_{idx:05d}.jpg"
        if src.exists():
            os.symlink(src, dst)

    output_ply = project_dir / "anysplat_output.ply"

    from pipeline.run_anysplat_subprocess import run_anysplat_subprocess

    import asyncio
    loop = asyncio.get_event_loop()

    # For chunked mode, use all selected frames (max_views just needs to be >= frame count)
    effective_max_views = len(selected_frames) if body.chunked else body.max_views

    start_time = time.time()
    try:
        n_gaussians = await loop.run_in_executor(
            _anysplat_executor,
            lambda: run_anysplat_subprocess(
                images_dir=temp_dir,
                output_ply=output_ply,
                max_views=effective_max_views,
                resolution=body.resolution,
                chunked=body.chunked,
                chunk_size=body.chunk_size,
                chunk_overlap=body.chunk_overlap,
            ),
        )
    except Exception as e:
        logger.error(f"AnySplat run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    duration_sec = round(time.time() - start_time, 1)

    # Copy PLY to scene.ply
    scene_ply = project_dir / "scene.ply"
    if output_ply.exists():
        shutil.copy2(str(output_ply), str(scene_ply))

    # Auto-cleanup: remove floaters, scale outliers, transparent gaussians
    cleanup_stats = None
    if scene_ply.exists():
        try:
            from pipeline.auto_cleanup import auto_cleanup
            cleanup_stats = auto_cleanup(scene_ply)
            n_gaussians = cleanup_stats["n_after"]
            logger.info(f"Auto-cleanup: {cleanup_stats}")
        except Exception as e:
            logger.warning(f"Auto-cleanup failed (non-fatal): {e}")

    # Generate SPZ + positions sidecar (best-effort)
    if scene_ply.exists():
        from utils.spz_convert import generate_spz_bundle
        generate_spz_bundle(project_dir)

    # Determine resolution used
    from pipeline.run_anysplat_subprocess import pick_resolution
    resolution_used = body.resolution if body.resolution > 0 else pick_resolution(body.max_views)

    # Update project in DB
    project.status = "ready"
    project.gaussian_count = n_gaussians
    project.reconstruction_mode = "anysplat"
    await db.commit()

    ply_url = f"/data/{project_id}/scene.ply"

    return AnySplatRunResult(
        status="ready",
        n_gaussians=n_gaussians,
        duration_sec=duration_sec,
        resolution_used=resolution_used,
        views_used=len(selected_frames),
        ply_url=ply_url,
        cleanup_stats=cleanup_stats,
    )
