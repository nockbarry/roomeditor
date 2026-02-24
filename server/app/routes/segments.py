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
    ClickSegment3DRequest,
    ClickSegmentRequest,
    DuplicateRequest,
    LightingRequest,
    MergeSegmentsRequest,
    RenameRequest,
    SegmentInfo,
    SegmentManifest,
    SegmentTransformRequest,
    SplitSegmentRequest,
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


def _write_ply_and_regenerate_spz(plydata, ply_path: Path):
    """Write PLY safely and regenerate SPZ + positions sidecar."""
    _write_ply_safe(plydata, ply_path)
    from utils.spz_convert import generate_spz_bundle
    generate_spz_bundle(ply_path.parent)


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


@router.post("/{project_id}/auto-segment-full", response_model=SegmentManifest)
async def auto_segment_full(
    project_id: str,
    max_frames: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """Run full auto segmentation pipeline: detect + multi-view assign in one shot."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    frames_dir = project_dir / "frames"
    ply_path = project_dir / "scene.ply"
    cameras_path = project_dir / "cameras.json"
    seg_path = project_dir / "segment_index_map.bin"

    if not frames_dir.exists() or not list(frames_dir.glob("frame_*.jpg")):
        raise HTTPException(status_code=400, detail="No frames found. Extract frames first.")
    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    import asyncio
    from app.routes.ws import broadcast_stream_update

    loop = asyncio.get_event_loop()

    async def _progress(frac, message):
        await broadcast_stream_update(project_id, {
            "type": "segment_progress",
            "progress": round(frac, 3),
            "message": message,
        })

    try:
        segments = await loop.run_in_executor(
            _segment_executor,
            lambda: _run_auto_segment_full(
                ply_path, frames_dir, seg_path, cameras_path, max_frames, loop, _progress
            ),
        )
    except Exception as e:
        logger.error(f"Auto-segment-full failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Read the manifest that was written by segment_scene
    manifest_path = project_dir / "segment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return _manifest_to_response(manifest)

    return {"segments": [], "total": 0, "primary_frame": None, "total_gaussians": 0, "unassigned_gaussians": 0}


def _run_auto_segment_full(ply_path, frames_dir, seg_path, cameras_path, max_frames, loop, progress_cb):
    """Run full segmentation pipeline in thread pool."""
    import asyncio
    from pipeline.segment_scene import segment_scene

    async def _async_progress(frac, msg):
        await progress_cb(frac, msg)

    def _sync_wrapper():
        return asyncio.run_coroutine_threadsafe(
            segment_scene(ply_path, frames_dir, seg_path, cameras_path, max_frames, _async_progress),
            loop,
        ).result()

    return _sync_wrapper()


@router.post("/{project_id}/delete-gaussians")
async def delete_gaussians_by_ids(
    project_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Delete specific gaussians by their indices (set opacity to -100)."""
    await _require_project(project_id, db)

    gaussian_ids = body.get("gaussian_ids", [])
    if not gaussian_ids:
        raise HTTPException(status_code=400, detail="No gaussian_ids provided")

    project_dir = _get_project_dir(project_id)
    ply_path = project_dir / "scene.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=400, detail="No scene.ply found")

    # Try in-memory path first
    from pipeline.scene_manager import get_cached_scene, DeleteOp
    scene = get_cached_scene(project_id)
    if scene and scene.data is not None:
        scene.apply_edit(DeleteOp(indices=gaussian_ids))
        return {"status": "ok", "removed_gaussians": len(gaussian_ids)}

    # Fallback: direct PLY I/O
    _save_checkpoint(project_dir, f"delete {len(gaussian_ids)} gaussians")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _delete_gaussians(ply_path, gaussian_ids),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "removed_gaussians": len(gaussian_ids)}


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


@router.post("/{project_id}/click-segment-3d", response_model=SegmentInfo)
async def click_segment_3d(
    project_id: str,
    body: ClickSegment3DRequest,
    db: AsyncSession = Depends(get_db),
):
    """Segment an object from a 3D point click.

    Projects the world-space point into all camera frames, picks the frame
    where the point is most central, then runs SAM2 click segmentation.
    """
    await _require_project(project_id, db)
    project_dir = _get_project_dir(project_id)

    cameras_path = project_dir / "cameras.json"
    if not cameras_path.exists():
        raise HTTPException(status_code=404, detail="cameras.json not found")

    import asyncio

    loop = asyncio.get_event_loop()

    try:
        segment = await loop.run_in_executor(
            _segment_executor,
            lambda: _run_click_segment_3d(project_dir, body.point),
        )
    except Exception as e:
        logger.error(f"Click-segment-3d failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return segment


def _run_click_segment_3d(project_dir: Path, point: list[float]) -> dict:
    """Project a 3D point into camera frames and run SAM2 on the best one."""
    from pipeline.segment_objects import click_segment_project

    cameras_path = project_dir / "cameras.json"
    cam_data = json.loads(cameras_path.read_text())
    cameras = cam_data["cameras"]
    resolution = cam_data.get("resolution", 448)

    pt = np.array(point, dtype=np.float64)
    best_frame = None
    best_score = -1.0
    best_u, best_v = 0, 0

    for cam in cameras:
        transform = np.array(cam["transform"], dtype=np.float64)
        if transform.shape[0] == 4:
            transform = transform[:3]

        R = transform[:, :3]
        t = transform[:, 3]

        # Camera-to-world -> world-to-camera
        R_inv = R.T
        t_inv = -R_inv @ t

        xyz_cam = R_inv @ pt + t_inv

        # Must be in front of camera
        if xyz_cam[2] <= 0.01:
            continue

        fx = cam.get("fx", resolution * 0.85)
        fy = cam.get("fy", fx)
        cx = cam.get("cx", resolution / 2)
        cy = cam.get("cy", resolution / 2)

        u = fx * xyz_cam[0] / xyz_cam[2] + cx
        v = fy * xyz_cam[1] / xyz_cam[2] + cy

        # Check in-bounds
        if u < 0 or u >= resolution or v < 0 or v >= resolution:
            continue

        # Score: prefer frames where the point is most central
        # Normalized distance from center (0 = center, 1 = edge)
        du = (u - cx) / (resolution / 2)
        dv = (v - cy) / (resolution / 2)
        centrality = 1.0 - np.sqrt(du * du + dv * dv)

        if centrality > best_score:
            best_score = centrality
            best_frame = cam["frame"]
            best_u = int(round(u))
            best_v = int(round(v))

    if best_frame is None:
        raise ValueError("3D point is not visible in any camera frame")

    logger.info(
        f"click-segment-3d: best frame={best_frame} at ({best_u}, {best_v}) "
        f"centrality={best_score:.3f}"
    )

    frames_dir = project_dir / "frames"
    return click_segment_project(frames_dir, project_dir, best_frame, best_u, best_v)


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

    # Try in-memory path first (instant), fall back to PLY I/O
    from pipeline.scene_manager import get_cached_scene, TransformOp
    scene = get_cached_scene(project_id)
    if scene and scene.data is not None:
        op = TransformOp(
            indices=segment["gaussian_ids"],
            translation=tuple(body.translation) if body.translation else None,
            rotation=tuple(body.rotation) if body.rotation else None,
            scale=tuple(body.scale) if body.scale else None,
        )
        scene.apply_edit(op)
        return {"status": "ok", "segment_id": segment_id, "n_gaussians_modified": len(segment["gaussian_ids"])}

    # Fallback: direct PLY I/O
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

    _write_ply_and_regenerate_spz(plydata, ply_path)


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

    # Try in-memory path first
    from pipeline.scene_manager import get_cached_scene, DeleteOp
    scene = get_cached_scene(project_id)
    if scene and scene.data is not None:
        scene.apply_edit(DeleteOp(indices=segment["gaussian_ids"]))
    else:
        # Fallback: direct PLY I/O
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

    _write_ply_and_regenerate_spz(plydata, ply_path)


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

    from utils.spz_convert import generate_spz_bundle
    generate_spz_bundle(project_dir)

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

    from utils.spz_convert import generate_spz_bundle
    generate_spz_bundle(project_dir)

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

    # Try in-memory path: use delete/undelete for visibility
    from pipeline.scene_manager import get_cached_scene, DeleteOp, UndeleteOp
    scene = get_cached_scene(project_id)
    if scene and scene.data is not None:
        if body.visible:
            scene.apply_edit(UndeleteOp(indices=segment["gaussian_ids"]))
        else:
            scene.apply_edit(DeleteOp(indices=segment["gaussian_ids"]))
    else:
        # Fallback: direct PLY I/O
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
    _write_ply_and_regenerate_spz(plydata, ply_path)


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
    _write_ply_and_regenerate_spz(PlyData([new_element], text=False), ply_path)

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
# Classify Segments (CLIP)
# ---------------------------------------------------------------------------

@router.post("/{project_id}/classify-segments", response_model=SegmentManifest)
async def classify_segments(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Classify segments using CLIP zero-shot classification."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest_path = project_dir / "segment_manifest.json"
    frames_dir = project_dir / "frames"

    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="No segments found")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        manifest = await loop.run_in_executor(
            _segment_executor,
            lambda: _classify_segments(manifest_path, frames_dir),
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _manifest_to_response(manifest)


def _classify_segments(manifest_path: Path, frames_dir: Path) -> dict:
    from pipeline.classify_segments import classify_segments
    return classify_segments(manifest_path, frames_dir)


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

    # Try in-memory path first
    from pipeline.scene_manager import get_cached_scene, TransformOp
    scene = get_cached_scene(project_id)
    if scene and scene.data is not None:
        op = TransformOp(
            indices=all_ids,
            translation=tuple(body.transform.translation) if body.transform.translation else None,
            rotation=tuple(body.transform.rotation) if body.transform.rotation else None,
            scale=tuple(body.transform.scale) if body.transform.scale else None,
        )
        scene.apply_edit(op)
        return {"status": "ok", "n_segments": len(body.segment_ids), "n_gaussians_modified": len(all_ids)}

    # Fallback: direct PLY I/O
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
# Merge Segments
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/merge", response_model=SegmentManifest)
async def merge_segments(
    project_id: str,
    body: MergeSegmentsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Merge multiple segments into one."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)

    if len(body.segment_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 segments to merge")

    _save_checkpoint(project_dir, f"merge {len(body.segment_ids)} segments")

    # Gather all gaussian_ids from the segments to merge
    merged_ids = []
    merged_area = 0
    segments_to_remove = []
    for seg_id in body.segment_ids:
        seg = next((s for s in manifest["segments"] if s["id"] == seg_id), None)
        if seg:
            merged_ids.extend(seg.get("gaussian_ids", []))
            merged_area += seg.get("area", 0)
            segments_to_remove.append(seg_id)

    if not merged_ids:
        raise HTTPException(status_code=400, detail="No gaussians to merge")

    # Remove old segments
    manifest["segments"] = [s for s in manifest["segments"] if s["id"] not in segments_to_remove]

    # Create merged segment
    existing_ids = [s["id"] for s in manifest["segments"]]
    new_id = max(existing_ids) + 1 if existing_ids else 1

    new_seg = {
        "id": new_id,
        "label": body.label or "merged",
        "area": merged_area,
        "bbox": [0, 0, 0, 0],
        "confidence": 1.0,
        "primary_frame": manifest.get("primary_frame", ""),
        "color": _random_color(),
        "gaussian_ids": merged_ids,
        "n_gaussians": len(merged_ids),
        "visible": True,
    }
    manifest["segments"].append(new_seg)
    manifest["total"] = len(manifest["segments"])
    _write_manifest(project_dir, manifest)

    # Regenerate segment index map
    _regenerate_index_map(project_dir, manifest)

    return _manifest_to_response(manifest)


# ---------------------------------------------------------------------------
# Split Segment
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/{segment_id}/split", response_model=SegmentManifest)
async def split_segment(
    project_id: str,
    segment_id: int,
    body: SplitSegmentRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Split a segment into N clusters using k-means on gaussian positions."""
    if body is None:
        body = SplitSegmentRequest()

    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    _save_checkpoint(project_dir, f"split segment {segment_id} into {body.n_clusters}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        new_segments = await loop.run_in_executor(
            _segment_executor,
            lambda: _split_segment(ply_path, manifest, segment, body.n_clusters),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _write_manifest(project_dir, manifest)
    _regenerate_index_map(project_dir, manifest)

    return _manifest_to_response(manifest)


def _split_segment(ply_path: Path, manifest: dict, segment: dict, n_clusters: int) -> list[dict]:
    """Split a segment using k-means clustering."""
    from plyfile import PlyData
    from scipy.cluster.vq import kmeans2

    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    ids = np.array(segment["gaussian_ids"])
    positions = np.column_stack([
        np.array(vertices["x"])[ids],
        np.array(vertices["y"])[ids],
        np.array(vertices["z"])[ids],
    ]).astype(np.float64)

    _, labels = kmeans2(positions, n_clusters, minit="points")

    # Remove original segment
    manifest["segments"] = [s for s in manifest["segments"] if s["id"] != segment["id"]]

    existing_ids = [s["id"] for s in manifest["segments"]]
    next_id = max(existing_ids) + 1 if existing_ids else 1

    new_segments = []
    for cluster_idx in range(n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_gids = ids[cluster_mask].tolist()
        if not cluster_gids:
            continue

        new_seg = {
            "id": next_id,
            "label": f"{segment.get('label', 'object')}_{cluster_idx}",
            "area": segment.get("area", 0) // n_clusters,
            "bbox": segment.get("bbox", [0, 0, 0, 0]),
            "confidence": segment.get("confidence", 0),
            "primary_frame": segment.get("primary_frame", ""),
            "color": _random_color(),
            "gaussian_ids": cluster_gids,
            "n_gaussians": len(cluster_gids),
            "visible": True,
        }
        manifest["segments"].append(new_seg)
        new_segments.append(new_seg)
        next_id += 1

    manifest["total"] = len(manifest["segments"])
    return new_segments


# ---------------------------------------------------------------------------
# Per-Segment Mesh Export
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/{segment_id}/extract-mesh")
async def extract_segment_mesh(
    project_id: str,
    segment_id: int,
    format: str = "glb",
    db: AsyncSession = Depends(get_db),
):
    """Extract mesh for a single segment."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        mesh_url = await loop.run_in_executor(
            _segment_executor,
            lambda: _extract_segment_mesh(ply_path, project_dir, project_id, segment, format),
        )
    except Exception as e:
        logger.error(f"Segment mesh extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"mesh_url": mesh_url}


def _extract_segment_mesh(ply_path: Path, project_dir: Path, project_id: str, segment: dict, format: str) -> str:
    """Create filtered PLY with only segment's gaussians, then extract mesh."""
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(str(ply_path))
    ids = np.array(segment["gaussian_ids"])
    filtered_data = plydata["vertex"].data[ids].copy()
    filtered_element = PlyElement.describe(filtered_data, "vertex")

    tmp_ply = project_dir / f"segment_{segment['id']}_filtered.ply"
    PlyData([filtered_element], text=False).write(str(tmp_ply))

    output_name = f"segment_{segment['id']}_mesh.{format}"
    output_path = project_dir / output_name

    try:
        from pipeline.extract_mesh import extract_mesh
        extract_mesh(str(tmp_ply), str(output_path), format=format)
    finally:
        if tmp_ply.exists():
            tmp_ply.unlink()

    return f"/data/{project_id}/{output_name}"


# ---------------------------------------------------------------------------
# Per-Segment PLY Export
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/{segment_id}/export-ply")
async def export_segment_ply(
    project_id: str,
    segment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Export a segment's gaussians as a standalone PLY file."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        ply_url = await loop.run_in_executor(
            _segment_executor,
            lambda: _export_segment_ply(ply_path, project_dir, project_id, segment),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ply_url": ply_url}


def _export_segment_ply(ply_path: Path, project_dir: Path, project_id: str, segment: dict) -> str:
    """Filter PLY to only include segment's gaussians."""
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(str(ply_path))
    ids = np.array(segment["gaussian_ids"])
    filtered_data = plydata["vertex"].data[ids].copy()
    filtered_element = PlyElement.describe(filtered_data, "vertex")

    output_name = f"segment_{segment['id']}.ply"
    output_path = project_dir / output_name
    PlyData([filtered_element], text=False).write(str(output_path))

    return f"/data/{project_id}/{output_name}"


# ---------------------------------------------------------------------------
# Lighting Adjustment
# ---------------------------------------------------------------------------

@router.put("/{project_id}/segments/{segment_id}/lighting")
async def adjust_lighting(
    project_id: str,
    segment_id: int,
    body: LightingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Adjust SH coefficients (brightness/color) for a segment."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    # Try in-memory path first
    from pipeline.scene_manager import get_cached_scene, LightingOp
    scene = get_cached_scene(project_id)
    if scene and scene.data is not None:
        scene.apply_edit(LightingOp(
            indices=segment["gaussian_ids"],
            brightness=body.brightness,
            color_tint=tuple(body.color_tint),
            sh_scale=body.sh_scale,
        ))
        return {"status": "ok"}

    # Fallback: direct PLY I/O
    _save_checkpoint(project_dir, f"lighting segment {segment_id}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _adjust_lighting(ply_path, segment["gaussian_ids"], body),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}


def _adjust_lighting(ply_path: Path, gaussian_ids: list[int], params: LightingRequest):
    """Modify SH DC coefficients (f_dc_0/1/2) to adjust brightness/color."""
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertices = plydata["vertex"]

    ids = np.array(gaussian_ids)

    # Modify DC SH coefficients (brightness and color tint)
    for ch, (attr, tint) in enumerate(zip(
        ["f_dc_0", "f_dc_1", "f_dc_2"],
        params.color_tint,
    )):
        if attr in vertices.data.dtype.names:
            vals = np.array(vertices[attr], dtype=np.float64)
            vals[ids] *= params.brightness * tint
            vertices[attr] = vals.astype(np.float32)

    # Scale higher-order SH bands
    if params.sh_scale != 1.0:
        for i in range(45):  # Up to SH degree 3
            attr = f"f_rest_{i}"
            if attr in vertices.data.dtype.names:
                vals = np.array(vertices[attr], dtype=np.float64)
                vals[ids] *= params.sh_scale
                vertices[attr] = vals.astype(np.float32)

    _write_ply_and_regenerate_spz(plydata, ply_path)


# ---------------------------------------------------------------------------
# Inpaint Remove
# ---------------------------------------------------------------------------

@router.post("/{project_id}/segments/{segment_id}/inpaint-remove")
async def inpaint_remove(
    project_id: str,
    segment_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Remove a segment and inpaint the hole."""
    await _require_project(project_id, db)

    project_dir = _get_project_dir(project_id)
    manifest = _read_manifest(project_dir)
    ply_path = project_dir / "scene.ply"
    segment = _get_segment(manifest, segment_id)

    if not segment.get("gaussian_ids"):
        raise HTTPException(status_code=400, detail="Segment has no assigned gaussians")

    _save_checkpoint(project_dir, f"inpaint-remove segment {segment_id}")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            _segment_executor,
            lambda: _inpaint_remove(ply_path, project_dir, manifest, segment),
        )
    except Exception as e:
        logger.error(f"Inpaint removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Remove segment from manifest
    manifest["segments"] = [s for s in manifest["segments"] if s["id"] != segment_id]
    manifest["total"] = len(manifest["segments"])
    _write_manifest(project_dir, manifest)

    return {"status": "ok", "removed_gaussians": len(segment["gaussian_ids"])}


def _inpaint_remove(ply_path: Path, project_dir: Path, manifest: dict, segment: dict):
    """Remove segment gaussians. Inpainting pipeline runs if available, otherwise just hides."""
    try:
        from pipeline.inpaint_scene import inpaint_scene
        inpaint_scene(ply_path, project_dir, segment)
    except ImportError:
        # Fallback: just set opacity to -100 (simple removal without inpainting)
        logger.info("Inpainting pipeline not available, using simple removal")
        _delete_gaussians(ply_path, segment["gaussian_ids"])


def _regenerate_index_map(project_dir: Path, manifest: dict):
    """Regenerate segment_index_map.bin from manifest."""
    ply_path = project_dir / "scene.ply"
    if not ply_path.exists():
        return

    total_gaussians = _get_ply_vertex_count(ply_path)
    index_map = np.zeros(total_gaussians, dtype=np.uint8)

    for list_idx, seg in enumerate(manifest.get("segments", [])):
        seg_value = list_idx + 1
        if seg_value > 255:
            break
        for gid in seg.get("gaussian_ids", []):
            if 0 <= gid < total_gaussians:
                index_map[gid] = seg_value

    map_path = project_dir / "segment_index_map.bin"
    map_path.write_bytes(index_map.tobytes())


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
            "semantic_confidence": s.get("semantic_confidence"),
        })
    return {
        "segments": segments,
        "total": len(segments),
        "primary_frame": manifest.get("primary_frame"),
        "total_gaussians": manifest.get("total_gaussians", 0),
        "unassigned_gaussians": manifest.get("unassigned_gaussians", 0),
    }
