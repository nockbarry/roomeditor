"""Real-time collaborative editing via WebSocket."""

import json
import logging
import uuid
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["collab"])

# Room state: project_id -> set of connected websockets
_collab_rooms: dict[str, dict[str, WebSocket]] = defaultdict(dict)
# Locked segments: project_id -> { segment_id: user_id }
_segment_locks: dict[str, dict[int, str]] = defaultdict(dict)


async def _broadcast(project_id: str, message: dict, exclude_user: str | None = None):
    """Broadcast a message to all users in a room."""
    room = _collab_rooms.get(project_id, {})
    dead = []
    for user_id, ws in room.items():
        if user_id == exclude_user:
            continue
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(user_id)
    for uid in dead:
        room.pop(uid, None)


def _get_presence(project_id: str) -> list[dict]:
    """Get list of connected users."""
    return [
        {"user_id": uid, "color": _user_color(uid)}
        for uid in _collab_rooms.get(project_id, {})
    ]


def _user_color(user_id: str) -> str:
    """Deterministic color from user ID."""
    h = hash(user_id) % 360
    return f"hsl({h}, 70%, 60%)"


@router.websocket("/ws/collab/{project_id}")
async def collab_ws(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time collaboration."""
    await websocket.accept()

    user_id = str(uuid.uuid4())[:8]
    _collab_rooms[project_id][user_id] = websocket

    # Notify existing users
    await _broadcast(project_id, {
        "type": "presence",
        "users": _get_presence(project_id),
    })

    # Send current lock state to new user
    await websocket.send_json({
        "type": "lock_state",
        "locks": {
            str(seg_id): lock_user
            for seg_id, lock_user in _segment_locks.get(project_id, {}).items()
        },
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "cursor_move":
                await _broadcast(project_id, {
                    "type": "cursor_move",
                    "user_id": user_id,
                    "position": data.get("position"),
                    "color": _user_color(user_id),
                }, exclude_user=user_id)

            elif msg_type == "select_segment":
                await _broadcast(project_id, {
                    "type": "select_segment",
                    "user_id": user_id,
                    "segment_id": data.get("segment_id"),
                }, exclude_user=user_id)

            elif msg_type == "transform_start":
                seg_id = data.get("segment_id")
                if seg_id is not None:
                    locks = _segment_locks[project_id]
                    if seg_id not in locks:
                        locks[seg_id] = user_id
                        await _broadcast(project_id, {
                            "type": "segment_locked",
                            "segment_id": seg_id,
                            "user_id": user_id,
                        })

            elif msg_type == "transform_end":
                seg_id = data.get("segment_id")
                if seg_id is not None:
                    locks = _segment_locks[project_id]
                    if locks.get(seg_id) == user_id:
                        del locks[seg_id]
                        await _broadcast(project_id, {
                            "type": "segment_unlocked",
                            "segment_id": seg_id,
                        })
                    # Notify all to refresh PLY
                    await _broadcast(project_id, {
                        "type": "ply_changed",
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Collab WS error: {e}")
    finally:
        # Clean up
        _collab_rooms[project_id].pop(user_id, None)
        if not _collab_rooms[project_id]:
            del _collab_rooms[project_id]

        # Release any locks held by this user
        locks = _segment_locks.get(project_id, {})
        to_release = [seg_id for seg_id, uid in locks.items() if uid == user_id]
        for seg_id in to_release:
            del locks[seg_id]

        # Notify remaining users
        await _broadcast(project_id, {
            "type": "presence",
            "users": _get_presence(project_id),
        })
