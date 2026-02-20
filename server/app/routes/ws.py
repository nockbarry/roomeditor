import asyncio
import base64
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session
from app.models import Job

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Simple in-memory store for active WebSocket connections per job
_job_connections: dict[str, list[WebSocket]] = {}

# Streaming session connections (project_id -> list of WebSockets)
_stream_connections: dict[str, list[WebSocket]] = {}


async def broadcast_job_progress(job_id: str, data: dict):
    """Send progress update to all WebSocket connections watching this job."""
    connections = _job_connections.get(job_id, [])
    dead = []
    for ws in connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


async def broadcast_training_preview(job_id: str, data: dict):
    """Send training preview image + metrics to all WebSocket connections."""
    connections = _job_connections.get(job_id, [])
    dead = []
    for ws in connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


async def broadcast_stream_update(project_id: str, data: dict):
    """Send streaming reconstruction update to all watching clients."""
    connections = _stream_connections.get(project_id, [])
    dead = []
    for ws in connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in _job_connections:
        _job_connections[job_id] = []
    _job_connections[job_id].append(websocket)

    try:
        # Send current job state immediately
        async with async_session() as db:
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                await websocket.send_json({
                    "job_id": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "error_message": job.error_message,
                })

        # Keep connection alive, waiting for client messages or disconnect
        while True:
            try:
                # Wait for ping/pong or client close
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _job_connections:
            _job_connections[job_id] = [
                ws for ws in _job_connections[job_id] if ws != websocket
            ]
            if not _job_connections[job_id]:
                del _job_connections[job_id]


@router.websocket("/ws/stream/{project_id}")
async def streaming_reconstruction_ws(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for streaming incremental reconstruction.

    Protocol:
    - Client sends frames as binary messages (JPEG bytes)
    - Client can send JSON text messages for control:
      {"action": "start", "config": {...}}
      {"action": "stop"}
    - Server sends JSON status updates:
      {"type": "status", "phase": "...", "frame_count": N, ...}
      {"type": "ply_update", "url": "/data/{project_id}/scene.ply", ...}
      {"type": "error", "message": "..."}
    """
    await websocket.accept()

    if project_id not in _stream_connections:
        _stream_connections[project_id] = []
    _stream_connections[project_id].append(websocket)

    from pipeline.streaming import (
        StreamingConfig, StreamingReconstructor,
        create_session, get_session, remove_session,
    )

    session = None
    project_dir = settings.data_dir / "projects" / project_id
    frames_dir = project_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    async def on_progress(frac, message):
        await websocket.send_json({
            "type": "progress",
            "progress": frac,
            "message": message,
        })

    async def on_ply_update(frame_count, training_step, n_gaussians, ply_path):
        await broadcast_stream_update(project_id, {
            "type": "ply_update",
            "frame_count": frame_count,
            "training_step": training_step,
            "num_gaussians": n_gaussians,
            "url": f"/data/{project_id}/scene.ply",
        })

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                # JSON control message
                data = json.loads(message["text"])
                action = data.get("action")

                if action == "start":
                    config = StreamingConfig(**data.get("config", {}))
                    session = create_session(
                        project_id=project_id,
                        project_dir=project_dir,
                        config=config,
                        on_progress=on_progress,
                        on_ply_update=on_ply_update,
                    )
                    await websocket.send_json({
                        "type": "status",
                        "phase": "waiting",
                        "message": "Streaming session started. Send frames as binary.",
                    })

                elif action == "stop":
                    if session:
                        remove_session(project_id)
                        session = None
                    await websocket.send_json({
                        "type": "status",
                        "phase": "done",
                        "message": "Streaming session stopped.",
                    })

                elif action == "ping":
                    await websocket.send_json({"type": "pong"})

            elif "bytes" in message:
                # Binary frame data (JPEG)
                if session is None:
                    session = get_session(project_id)
                    if session is None:
                        # Auto-create session with defaults
                        session = create_session(
                            project_id=project_id,
                            project_dir=project_dir,
                            on_progress=on_progress,
                            on_ply_update=on_ply_update,
                        )

                # Save frame to disk
                frame_idx = session.state.frame_count
                frame_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
                with open(frame_path, "wb") as f:
                    f.write(message["bytes"])

                # Process frame
                try:
                    status = await session.add_frame(frame_path)
                    await websocket.send_json({
                        "type": "status",
                        **status,
                    })
                except Exception as e:
                    logger.error(f"Stream frame processing error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

    except WebSocketDisconnect:
        logger.info(f"Stream client disconnected: {project_id}")
    except Exception as e:
        logger.error(f"Stream WebSocket error: {e}", exc_info=True)
    finally:
        if project_id in _stream_connections:
            _stream_connections[project_id] = [
                ws for ws in _stream_connections[project_id] if ws != websocket
            ]
            # If no more clients watching, clean up the streaming session
            if not _stream_connections[project_id]:
                del _stream_connections[project_id]
                try:
                    remove_session(project_id)
                    logger.info(f"Cleaned up streaming session: {project_id}")
                except Exception:
                    pass
