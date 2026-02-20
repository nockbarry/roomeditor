"""Segmentation endpoints — SAM2-based object detection and gaussian assignment."""

import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Project
from app.schemas import (
    BatchTransformRequest,
    ClickSegmentRequest,
    DuplicateRequest,
    RenameRequest,
    SegmentInfo,
    SegmentManifest,
    SegmentTransformRequest,
    VisibilityRequest,
)

logger = logging.getLogger(__name__)
_segment_executor = ThreadPoolExecutor(max_workers=1)

router = APIRouter(prefix="/api/projects", tags=["segments"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_dir(project_id: str) -> Path:
    return settings.data_dir / "projects" / project_id


def _read_manifest(project_dir: Path) -> dict:
    manifest_path = project_dir / "segment_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="No segments found")
    return json.loads(manifest_path.read_text())


def _write_manifest(project_dir: Path, manifest: dict):
    manifest_path = project_dir / "segment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _get_segment(manifest: dict, segment_id: int) -> dict:
    segment = next((s for s in manifest["segments"] if s["id"] == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    return segment


async def _require_project(project_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _save_checkpoint(project_dir: Path, label: str):
    """Save checkpoint before mutating operations."""
    try:
        from pipeline.undo import save_checkpoint
        save_checkpoint(project_dir, label)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def _get_ply_vertex_count(ply_path: Path) -> int:
    """Parse just the PLY header text to extract vertex count without loading binary data."""
    with open(ply_path, "rb") as f:
        # Read enough for the header (typically < 4KB)
        header_bytes = f.read(65536)
    header_text = header_bytes.split(b"end_header")[0].decode("ascii", errors="replace")
    for line in header_text.split("\n"):
        line = line.strip()
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    raise ValueError(f"No vertex count found in PLY header: {ply_path}")


def _write_ply_safe(plydata, ply_path: Path):
    """Write PlyData to a temp file then rename, to avoid mmap conflicts."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=str(ply_path.parent))
    try:
        import os
        os.close(tmp_fd)
        plydata.write(tmp_path)
        shutil.move(tmp_path, str(ply_path))
    except Exception:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# SAM2 Segmentation
# ---------------------------------------------------------------------------

@router.post("/{project_id}/auto-segment", response_model=SegmentManifest)
async def auto_segment(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Run SAM2 automatic segmentation on key frames."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    frames_dir = project_dir / "frames"

    if not frames_dir.exists() or not list(frames_dir.glob("frame_*.jpg")):
        raise HTTPException(status_code=400, detail="No frames found. Extract frames first.")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        manifest = await loop.run_in_executor(
            _segment_executor,
            lambda: _run_auto_segment(frames_dir, project_dir),
        )
    except Exception as e:
        logger.error(f"Auto-segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _manifest_to_response(manifest)


def _run_auto_segment(frames_dir: Path, project_dir: Path) -> dict:
    """Run auto segmentation (called in thread pool)."""
    from pipeline.segment_objects import auto_segment_project
    return auto_segment_project(frames_dir, project_dir)


@router.post("/{project_id}/click-segment", response_model=SegmentInfo)
async def click_segment(
    project_id: str,
    body: ClickSegmentRequest,
    db: AsyncSession = Depends(get_db),
):
    """Segment an object at a click point."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    frames_dir = project_dir / "frames"
    frame_path = frames_dir / body.frame

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {body.frame}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        segment = await loop.run_in_executor(
            _segment_executor,
            lambda: _run_click_segment(frames_dir, project_dir, body.frame, body.x, body.y),
        )
    except Exception as e:
        logger.error(f"Click segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return segment


def _run_click_segment(frames_dir, project_dir, frame, x, y) -> dict:
    from pipeline.segment_objects import click_segment_project
    return click_segment_project(frames_dir, project_dir, frame, x, y)


# ---------------------------------------------------------------------------
# Segment Queries
# ---------------------------------------------------------------------------

@router.get("/{project_id}/segments", response_model=SegmentManifest)
async def get_segments(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get segment manifest."""
    await _require_project(project_id, db)

    manifest_path = _get_project_dir(project_id) / "segment_manifest.json"
    if not manifest_path.exists():
        return {"segments": [], "total": 0, "primary_frame": None, "total_gaussians": 0, "unassigned_gaussians": 0}

    manifest = json.loads(manifest_path.read_text())
    return _manifest_to_response(manifest)


@router.get("/{project_id}/segment-index-map")
async def get_segment_index_map(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return a binary Uint8Array mapping gaussian index → segment list index + 1 (0 = unassigned).

    4.6M gaussians = 4.6MB binary response, ~1s transfer.
    """
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest_path = project_dir / "segment_manifest.json"
    ply_path = project_dir / "scene.ply"

    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="No segments found")
    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    manifest = json.loads(manifest_path.read_text())
    total_gaussians = _get_ply_vertex_count(ply_path)

    # Build the index map: 0 = unassigned, 1+ = segment list index + 1
    index_map = np.zeros(total_gaussians, dtype=np.uint8)
    for list_idx, seg in enumerate(manifest.get("segments", [])):
        seg_value = list_idx + 1  # 1-based
        if seg_value > 255:
            break  # Uint8 can only hold 255 segments
        for gid in seg.get("gaussian_ids", []):
            if 0 <= gid < total_gaussians:
                index_map[gid] = seg_value

    return Response(
        content=index_map.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Vertex-Count": str(total_gaussians)},
    )


@router.post("/{project_id}/assign-gaussians", response_model=SegmentManifest)
async def assign_gaussians(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Backproject 2D segment masks to 3D gaussians."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    frames_dir = project_dir / "frames"

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        manifest = await loop.run_in_executor(
            _segment_executor,
            lambda: _run_assign(project_dir, frames_dir),
        )
    except Exception as e:
        logger.error(f"Gaussian assignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _manifest_to_response(manifest)


def _run_assign(project_dir, frames_dir):
    from pipeline.segment_objects import assign_gaussians_to_segments
    return assign_gaussians_to_segments(project_dir, frames_dir)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

@router.put("/{project_id}/segments/{segment_id}/transform")
async def transform_segment(
    project_id: str,
    segment_id: int,
    body: SegmentTransformRequest,
    db: AsyncSession = Depends(get_db),
):
    """Transform (move/rotate/scale) a segment's gaussians in the PLY."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians. Run assign-gaussians first.")

    _save_checkpoint(project_dir, f"transform segment {segment_id}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _apply_transform(ply_path, segment["gaussian_ids"], body),
        )
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "segment_id": segment_id, "n_gaussians_modified": len(segment["gaussian_ids"])}


def _apply_transform(ply_path: Path, gaussian_ids: list[int], transform: SegmentTransformRequest):
    """Apply translation/rotation/scale to specific gaussians in a PLY file."""
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    ids = np.array(gaussian_ids)

    x = np.array(vertices["x"], dtype=np.float64)
    y = np.array(vertices["y"], dtype=np.float64)
    z = np.array(vertices["z"], dtype=np.float64)

    # Compute centroid for rotation/scale pivot
    cx, cy, cz = x[ids].mean(), y[ids].mean(), z[ids].mean()

    if transform.rotation:
        rx, ry, rz = transform.rotation  # euler angles in degrees
        if rx != 0 or ry != 0 or rz != 0:
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler("xyz", [rx, ry, rz], degrees=True)
            rot_matrix = rot.as_matrix()

            # Rotate positions relative to centroid
            positions = np.column_stack([x[ids] - cx, y[ids] - cy, z[ids] - cz])
            rotated = positions @ rot_matrix.T
            x[ids] = rotated[:, 0] + cx
            y[ids] = rotated[:, 1] + cy
            z[ids] = rotated[:, 2] + cz

            # Compose rotation with existing per-gaussian quaternions
            quat_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
            if all(n in vertices.data.dtype.names for n in quat_names):
                # Read existing quaternions (w, x, y, z format for 3DGS)
                qw = np.array(vertices["rot_0"], dtype=np.float64)
                qx = np.array(vertices["rot_1"], dtype=np.float64)
                qy = np.array(vertices["rot_2"], dtype=np.float64)
                qz = np.array(vertices["rot_3"], dtype=np.float64)

                existing_quats = np.column_stack([qx[ids], qy[ids], qz[ids], qw[ids]])  # scipy: (x,y,z,w)
                existing_rots = R.from_quat(existing_quats)
                new_rots = rot * existing_rots
                new_quats = new_rots.as_quat()  # (x,y,z,w)

                qw[ids] = new_quats[:, 3]
                qx[ids] = new_quats[:, 0]
                qy[ids] = new_quats[:, 1]
                qz[ids] = new_quats[:, 2]

                vertices["rot_0"] = qw.astype(np.float32)
                vertices["rot_1"] = qx.astype(np.float32)
                vertices["rot_2"] = qy.astype(np.float32)
                vertices["rot_3"] = qz.astype(np.float32)

    if transform.scale:
        sx, sy, sz = transform.scale
        # Recompute centroid after potential rotation
        cx, cy, cz = x[ids].mean(), y[ids].mean(), z[ids].mean()
        x[ids] = (x[ids] - cx) * sx + cx
        y[ids] = (y[ids] - cy) * sy + cy
        z[ids] = (z[ids] - cz) * sz + cz

        # Also scale the gaussian scales (log-space)
        for attr in ["scale_0", "scale_1", "scale_2"]:
            if attr in vertices.data.dtype.names:
                vals = np.array(vertices[attr])
                scale_factor = [sx, sy, sz]["scale_0 scale_1 scale_2".split().index(attr)]
                if scale_factor > 0:
                    vals[ids] += np.log(scale_factor)
                vertices[attr] = vals

    if transform.translation:
        tx, ty, tz = transform.translation
        x[ids] += tx
        y[ids] += ty
        z[ids] += tz

    vertices["x"] = x.astype(np.float32)
    vertices["y"] = y.astype(np.float32)
    vertices["z"] = z.astype(np.float32)

    _write_ply_safe(plydata, ply_path)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{project_id}/segments/{segment_id}")
async def delete_segment(
    project_id: str,
    segment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Remove a segment's gaussians from the PLY (set opacity to 0)."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    _save_checkpoint(project_dir, f"delete segment {segment_id}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _delete_gaussians(ply_path, segment["gaussian_ids"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Remove segment from manifest
    manifest["segments"] = [s for s in manifest["segments"] if s["id"] != segment_id]
    manifest["total"] = len(manifest["segments"])
    _write_manifest(project_dir, manifest)

    return {"status": "ok", "removed_gaussians": len(segment["gaussian_ids"])}


def _delete_gaussians(ply_path: Path, gaussian_ids: list[int]):
    """Set opacity to -infinity (invisible) for specified gaussians."""
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    ids = np.array(gaussian_ids)

    if "opacity" in vertices.data.dtype.names:
        opacities = np.array(vertices["opacity"])
        opacities[ids] = -100.0
        vertices["opacity"] = opacities

    _write_ply_safe(plydata, ply_path)


# ---------------------------------------------------------------------------
# Undo / Restore
# ---------------------------------------------------------------------------

@router.post("/{project_id}/undo")
async def undo(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Restore the most recent checkpoint."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)

    try:
        from pipeline.undo import restore_checkpoint
        info = restore_checkpoint(project_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok", **info}


@router.post("/{project_id}/restore-original")
async def restore_original(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Restore original AnySplat output."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    original_ply = project_dir / "anysplat_output.ply"
    scene_ply = project_dir / "scene.ply"
    manifest_path = project_dir / "segment_manifest.json"

    if not original_ply.exists():
        raise HTTPException(status_code=400, detail="No original AnySplat output found")

    shutil.copy2(str(original_ply), str(scene_ply))

    # Clear segment manifest
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for seg in manifest.get("segments", []):
            seg.pop("gaussian_ids", None)
            seg["n_gaussians"] = 0
            seg.pop("saved_opacities", None)
            seg["visible"] = True
        _write_manifest(project_dir, manifest)

    # Clear undo stack
    undo_dir = project_dir / "undo"
    if undo_dir.exists():
        shutil.rmtree(str(undo_dir))

    return {"status": "ok"}


@router.get("/{project_id}/undo-stack")
async def get_undo_stack(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get available undo checkpoints."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)

    from pipeline.undo import list_checkpoints
    checkpoints = list_checkpoints(project_dir)

    return {"checkpoints": checkpoints, "count": len(checkpoints)}


# ---------------------------------------------------------------------------
# Visibility Toggle
# ---------------------------------------------------------------------------

@router.put("/{project_id}/segments/{segment_id}/visibility")
async def toggle_visibility(
    project_id: str,
    segment_id: int,
    body: VisibilityRequest,
    db: AsyncSession = Depends(get_db),
):
    """Toggle segment visibility by manipulating opacity values."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    _save_checkpoint(project_dir, f"{'show' if body.visible else 'hide'} segment {segment_id}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _toggle_visibility(ply_path, segment, body.visible),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Update manifest
    segment["visible"] = body.visible
    _write_manifest(project_dir, manifest)

    return {"status": "ok", "segment_id": segment_id, "visible": body.visible}


def _toggle_visibility(ply_path: Path, segment: dict, visible: bool):
    """Hide: save opacities, set to -100. Show: restore saved opacities."""
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    if "opacity" not in vertices.data.dtype.names:
        return

    ids = np.array(segment["gaussian_ids"])
    opacities = np.array(vertices["opacity"])

    if not visible:
        # Save current opacities before hiding
        segment["saved_opacities"] = opacities[ids].tolist()
        opacities[ids] = -100.0
    else:
        # Restore saved opacities
        saved = segment.get("saved_opacities")
        if saved and len(saved) == len(ids):
            opacities[ids] = np.array(saved, dtype=np.float32)
            segment.pop("saved_opacities", None)
        else:
            # No saved opacities — set to a reasonable default
            opacities[ids] = 0.0
            segment.pop("saved_opacities", None)

    vertices["opacity"] = opacities
    _write_ply_safe(plydata, ply_path)


# ---------------------------------------------------------------------------
# Duplicate / Clone
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/{segment_id}/duplicate")
async def duplicate_segment(
    project_id: str,
    segment_id: int,
    body: DuplicateRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a segment's gaussians with an offset."""
    if body is None:
        body = DuplicateRequest()

    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    _save_checkpoint(project_dir, f"duplicate segment {segment_id}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        new_segment = await loop.run_in_executor(
            _segment_executor,
            lambda: _duplicate_gaussians(ply_path, manifest, segment, body.offset),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _write_manifest(project_dir, manifest)

    return {
        "status": "ok",
        "new_segment": {
            "id": new_segment["id"],
            "label": new_segment["label"],
            "n_gaussians": new_segment.get("n_gaussians", len(new_segment.get("gaussian_ids", []))),
        },
    }


def _duplicate_gaussians(ply_path: Path, manifest: dict, segment: dict, offset: list[float]) -> dict:
    """Copy a segment's gaussians, offset positions, append to PLY."""
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]
    old_count = len(vertices.data)

    ids = np.array(segment["gaussian_ids"])
    cloned_data = vertices.data[ids].copy()

    # Offset positions
    ox, oy, oz = offset
    x = np.array(cloned_data["x"], dtype=np.float64)
    y = np.array(cloned_data["y"], dtype=np.float64)
    z = np.array(cloned_data["z"], dtype=np.float64)
    cloned_data["x"] = (x + ox).astype(np.float32)
    cloned_data["y"] = (y + oy).astype(np.float32)
    cloned_data["z"] = (z + oz).astype(np.float32)

    # Merge old + new
    merged = np.concatenate([vertices.data, cloned_data])
    new_element = PlyElement.describe(merged, "vertex")
    _write_ply_safe(PlyData([new_element], text=False), ply_path)

    # Create new segment in manifest
    new_ids = list(range(old_count, old_count + len(ids)))
    existing_ids = [s["id"] for s in manifest["segments"]]
    new_id = max(existing_ids) + 1 if existing_ids else 1

    new_segment = {
        "id": new_id,
        "label": segment.get("label", f"object_{segment['id']}") + " (copy)",
        "area": segment.get("area", 0),
        "bbox": segment.get("bbox", [0, 0, 0, 0]),
        "confidence": segment.get("confidence", 0),
        "primary_frame": segment.get("primary_frame", ""),
        "color": _random_color(),
        "gaussian_ids": new_ids,
        "n_gaussians": len(new_ids),
        "visible": True,
    }
    manifest["segments"].append(new_segment)
    manifest["total"] = len(manifest["segments"])
    manifest["total_gaussians"] = manifest.get("total_gaussians", 0) + len(new_ids)

    return new_segment


def _random_color() -> list[int]:
    """Generate a random distinct color for segments."""
    import random
    return [random.randint(60, 230), random.randint(60, 230), random.randint(60, 230)]


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------

@router.put("/{project_id}/segments/{segment_id}/rename")
async def rename_segment(
    project_id: str,
    segment_id: int,
    body: RenameRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rename a segment."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    segment = _get_segment(manifest, segment_id)

    segment["label"] = body.label
    _write_manifest(project_dir, manifest)

    return {"status": "ok", "segment_id": segment_id, "label": body.label}


# ---------------------------------------------------------------------------
# Background Segment
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/create-background")
async def create_background_segment(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Create a segment for all unassigned gaussians."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    total = len(plydata["vertex"].data)

    # Collect all assigned gaussian IDs
    assigned = set()
    for seg in manifest.get("segments", []):
        for gid in seg.get("gaussian_ids", []):
            assigned.add(gid)

    unassigned = [i for i in range(total) if i not in assigned]

    if not unassigned:
        raise HTTPException(status_code=400, detail="No unassigned gaussians")

    existing_ids = [s["id"] for s in manifest["segments"]]
    new_id = max(existing_ids) + 1 if existing_ids else 1

    bg_segment = {
        "id": new_id,
        "label": "Background",
        "area": 0,
        "bbox": [0, 0, 0, 0],
        "confidence": 1.0,
        "primary_frame": manifest.get("primary_frame", ""),
        "color": [100, 100, 100],
        "gaussian_ids": unassigned,
        "n_gaussians": len(unassigned),
        "visible": True,
    }
    manifest["segments"].append(bg_segment)
    manifest["total"] = len(manifest["segments"])
    manifest["unassigned_gaussians"] = 0
    _write_manifest(project_dir, manifest)

    return {
        "status": "ok",
        "segment": {
            "id": new_id,
            "label": "Background",
            "n_gaussians": len(unassigned),
        },
    }


# ---------------------------------------------------------------------------
# Batch Transform
# ---------------------------------------------------------------------------

@router.put("/{project_id}/segments/batch-transform")
async def batch_transform(
    project_id: str,
    body: BatchTransformRequest,
    db: AsyncSession = Depends(get_db),
):
    """Transform multiple segments at once (single PLY read/write)."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    # Gather all gaussian IDs from specified segments
    all_ids = []
    for seg_id in body.segment_ids:
        seg = next((s for s in manifest["segments"] if s["id"] == seg_id), None)
        if seg and seg.get("gaussian_ids"):
            all_ids.extend(seg["gaussian_ids"])

    if not all_ids:
        raise HTTPException(status_code=400, detail="No gaussians to transform")

    _save_checkpoint(project_dir, f"batch transform {len(body.segment_ids)} segments")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _apply_transform(ply_path, all_ids, body.transform),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "n_segments": len(body.segment_ids), "n_gaussians_modified": len(all_ids)}


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _manifest_to_response(manifest: dict) -> dict:
    """Convert raw manifest to API response format."""
    segments = []
    for s in manifest.get("segments", []):
        segments.append({
            "id": s["id"],
            "label": s.get("label", f"object_{s['id']}"),
            "area": s.get("area", 0),
            "bbox": s.get("bbox", [0, 0, 0, 0]),
            "confidence": s.get("confidence", 0),
            "primary_frame": s.get("primary_frame", ""),
            "color": s.get("color", [128, 128, 128]),
            "n_gaussians": s.get("n_gaussians", len(s.get("gaussian_ids", []))),
            "click_point": s.get("click_point"),
            "visible": s.get("visible", True),
        })
    return {
        "segments": segments,
        "total": len(segments),
        "primary_frame": manifest.get("primary_frame"),
        "total_gaussians": manifest.get("total_gaussians", 0),
        "unassigned_gaussians": manifest.get("unassigned_gaussians", 0),
    }
