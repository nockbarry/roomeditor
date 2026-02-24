"""Scene editing endpoints — in-memory editing with delta undo/redo."""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Project

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["editing"])


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class TransformRequest(BaseModel):
    translation: list[float] | None = None
    rotation: list[float] | None = None
    scale: list[float] | None = None


class EditRequest(BaseModel):
    """Polymorphic edit operation."""
    type: Literal["transform", "delete", "undelete", "lighting", "property"]
    indices: list[int]
    # Transform fields
    translation: list[float] | None = None
    rotation: list[float] | None = None
    scale: list[float] | None = None
    # Lighting fields
    brightness: float = 1.0
    color_tint: list[float] = [1.0, 1.0, 1.0]
    sh_scale: float = 1.0
    # Property fields
    attr: str | None = None
    values: list[float] | None = None


class BoxQuery(BaseModel):
    min: list[float]  # [x, y, z]
    max: list[float]  # [x, y, z]
    exclude_deleted: bool = True


class SphereQuery(BaseModel):
    center: list[float]  # [x, y, z]
    radius: float
    exclude_deleted: bool = True


class DeleteRegionRequest(BaseModel):
    """Delete gaussians in/outside a box or sphere."""
    shape: Literal["box", "sphere"]
    # Box params
    min: list[float] | None = None
    max: list[float] | None = None
    # Sphere params
    center: list[float] | None = None
    radius: float | None = None
    # Mode
    mode: Literal["inside", "outside"] = "inside"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _require_project(project_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _get_project_dir(project_id: str):
    return settings.data_dir / "projects" / project_id


def _get_scene(project_id: str):
    """Get the scene manager, loading if needed."""
    from pipeline.scene_manager import get_scene
    project_dir = _get_project_dir(project_id)
    return get_scene(project_id, project_dir)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{project_id}/scene/load")
async def load_scene(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Load scene into memory. Returns gaussian count and bounds."""
    await _require_project(project_id, db)
    try:
        scene = _get_scene(project_id)
        info = scene.get_history()
        return {
            "status": "ok",
            "n_gaussians": scene.n_gaussians,
            "n_deleted": int(scene.deleted.sum()) if scene.deleted is not None else 0,
            **info,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="No scene.ply found")


@router.post("/{project_id}/scene/save")
async def save_scene(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Flush in-memory scene to PLY + regenerate SPZ."""
    await _require_project(project_id, db)

    from pipeline.scene_manager import get_cached_scene
    scene = get_cached_scene(project_id)
    if not scene:
        raise HTTPException(status_code=400, detail="Scene not loaded in memory")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(None, scene.save)
    except Exception as e:
        logger.error(f"Scene save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", **result}


@router.post("/{project_id}/scene/edit")
async def edit_scene(
    project_id: str,
    body: EditRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply an edit operation to the in-memory scene."""
    await _require_project(project_id, db)

    from pipeline.scene_manager import get_cached_scene, TransformOp, DeleteOp, UndeleteOp, LightingOp, PropertyEditOp
    import numpy as np

    scene = get_cached_scene(project_id)
    if not scene or scene.data is None:
        # Auto-load if not cached
        scene = _get_scene(project_id)

    if not body.indices:
        raise HTTPException(status_code=400, detail="No indices provided")

    if body.type == "transform":
        op = TransformOp(
            indices=body.indices,
            translation=tuple(body.translation) if body.translation else None,
            rotation=tuple(body.rotation) if body.rotation else None,
            scale=tuple(body.scale) if body.scale else None,
        )
    elif body.type == "delete":
        op = DeleteOp(indices=body.indices)
    elif body.type == "undelete":
        op = UndeleteOp(indices=body.indices)
    elif body.type == "lighting":
        op = LightingOp(
            indices=body.indices,
            brightness=body.brightness,
            color_tint=tuple(body.color_tint),
            sh_scale=body.sh_scale,
        )
    elif body.type == "property":
        if not body.attr or body.values is None:
            raise HTTPException(status_code=400, detail="attr and values required for property edit")
        op = PropertyEditOp(
            indices=body.indices,
            attr=body.attr,
            values=np.array(body.values, dtype=np.float32),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown edit type: {body.type}")

    try:
        result = scene.apply_edit(op)
    except Exception as e:
        logger.error(f"Edit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", **result}


@router.post("/{project_id}/scene/undo")
async def undo_edit(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Undo the last edit."""
    await _require_project(project_id, db)

    from pipeline.scene_manager import get_cached_scene
    scene = get_cached_scene(project_id)
    if not scene:
        raise HTTPException(status_code=400, detail="Scene not loaded in memory")

    result = scene.undo()
    if result is None:
        raise HTTPException(status_code=400, detail="Nothing to undo")

    return {"status": "ok", **result}


@router.post("/{project_id}/scene/redo")
async def redo_edit(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Redo the last undone edit."""
    await _require_project(project_id, db)

    from pipeline.scene_manager import get_cached_scene
    scene = get_cached_scene(project_id)
    if not scene:
        raise HTTPException(status_code=400, detail="Scene not loaded in memory")

    result = scene.redo()
    if result is None:
        raise HTTPException(status_code=400, detail="Nothing to redo")

    return {"status": "ok", **result}


@router.get("/{project_id}/scene/history")
async def get_history(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get undo/redo stack labels and dirty state."""
    await _require_project(project_id, db)

    from pipeline.scene_manager import get_cached_scene
    scene = get_cached_scene(project_id)
    if not scene:
        return {"undo": [], "redo": [], "undo_count": 0, "redo_count": 0, "dirty": False}

    return scene.get_history()


@router.post("/{project_id}/scene/query-box")
async def query_box(
    project_id: str,
    body: BoxQuery,
    db: AsyncSession = Depends(get_db),
):
    """Return gaussian indices within an AABB."""
    await _require_project(project_id, db)
    scene = _get_scene(project_id)

    indices = scene.query_box(
        tuple(body.min), tuple(body.max),
        exclude_deleted=body.exclude_deleted,
    )

    return {
        "indices": indices.tolist(),
        "count": len(indices),
    }


@router.post("/{project_id}/scene/query-sphere")
async def query_sphere(
    project_id: str,
    body: SphereQuery,
    db: AsyncSession = Depends(get_db),
):
    """Return gaussian indices within a sphere."""
    await _require_project(project_id, db)
    scene = _get_scene(project_id)

    indices = scene.query_sphere(
        tuple(body.center), body.radius,
        exclude_deleted=body.exclude_deleted,
    )

    return {
        "indices": indices.tolist(),
        "count": len(indices),
    }


@router.post("/{project_id}/scene/query-radius")
async def query_radius(
    project_id: str,
    body: SphereQuery,
    db: AsyncSession = Depends(get_db),
):
    """Return gaussian indices near a point (brush tool)."""
    await _require_project(project_id, db)
    scene = _get_scene(project_id)

    indices = scene.query_radius(
        tuple(body.center), body.radius,
        exclude_deleted=body.exclude_deleted,
    )

    return {
        "indices": indices.tolist(),
        "count": len(indices),
    }


@router.post("/{project_id}/scene/delete-region")
async def delete_region(
    project_id: str,
    body: DeleteRegionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete gaussians inside or outside a box/sphere."""
    await _require_project(project_id, db)

    from pipeline.scene_manager import get_cached_scene, DeleteOp
    import numpy as np

    scene = get_cached_scene(project_id)
    if not scene or scene.data is None:
        scene = _get_scene(project_id)

    # Get indices in the region
    if body.shape == "box":
        if not body.min or not body.max:
            raise HTTPException(status_code=400, detail="min and max required for box")
        inside = scene.query_box(tuple(body.min), tuple(body.max))
    elif body.shape == "sphere":
        if not body.center or body.radius is None:
            raise HTTPException(status_code=400, detail="center and radius required for sphere")
        inside = scene.query_sphere(tuple(body.center), body.radius)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown shape: {body.shape}")

    if body.mode == "inside":
        target_indices = inside
    else:
        # Delete outside: all non-deleted gaussians NOT in the region
        all_active = np.where(~scene.deleted)[0].astype(np.uint32)
        inside_set = set(inside.tolist())
        target_indices = np.array(
            [i for i in all_active if i not in inside_set],
            dtype=np.uint32,
        )

    if len(target_indices) == 0:
        return {"status": "ok", "n_affected": 0, "undo_count": len(scene.undo_stack), "redo_count": len(scene.redo_stack)}

    result = scene.apply_edit(DeleteOp(indices=target_indices.tolist()))
    return {"status": "ok", **result}
