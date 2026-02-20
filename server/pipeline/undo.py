"""Undo system — checkpoint save/restore for scene.ply and segment_manifest.json."""

import json
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CHECKPOINTS = 10


def save_checkpoint(project_dir: Path, label: str = "") -> str:
    """Save current scene.ply + segment_manifest.json as a checkpoint.

    Returns the checkpoint ID (timestamp-based).
    """
    undo_dir = project_dir / "undo"
    undo_dir.mkdir(exist_ok=True)

    checkpoint_id = f"{int(time.time() * 1000)}"
    cp_dir = undo_dir / checkpoint_id
    cp_dir.mkdir()

    # Copy scene.ply
    scene_ply = project_dir / "scene.ply"
    if scene_ply.exists():
        shutil.copy2(str(scene_ply), str(cp_dir / "scene.ply"))

    # Copy segment manifest
    manifest_path = project_dir / "segment_manifest.json"
    if manifest_path.exists():
        shutil.copy2(str(manifest_path), str(cp_dir / "segment_manifest.json"))

    # Save metadata
    meta = {"label": label, "timestamp": time.time()}
    (cp_dir / "meta.json").write_text(json.dumps(meta))

    logger.info(f"Saved checkpoint {checkpoint_id}: {label}")

    # Prune old checkpoints (keep max N)
    _prune_checkpoints(undo_dir)

    return checkpoint_id


def restore_checkpoint(project_dir: Path, checkpoint_id: str | None = None) -> dict:
    """Restore the most recent (or specified) checkpoint.

    Returns info about the restored checkpoint.
    """
    undo_dir = project_dir / "undo"
    if not undo_dir.exists():
        raise FileNotFoundError("No checkpoints available")

    if checkpoint_id:
        cp_dir = undo_dir / checkpoint_id
    else:
        # Get most recent
        checkpoints = sorted(undo_dir.iterdir(), key=lambda p: p.name, reverse=True)
        checkpoints = [c for c in checkpoints if c.is_dir()]
        if not checkpoints:
            raise FileNotFoundError("No checkpoints available")
        cp_dir = checkpoints[0]
        checkpoint_id = cp_dir.name

    if not cp_dir.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")

    # Restore scene.ply
    cp_ply = cp_dir / "scene.ply"
    if cp_ply.exists():
        shutil.copy2(str(cp_ply), str(project_dir / "scene.ply"))

    # Restore manifest
    cp_manifest = cp_dir / "segment_manifest.json"
    if cp_manifest.exists():
        shutil.copy2(str(cp_manifest), str(project_dir / "segment_manifest.json"))

    # Read metadata
    meta_path = cp_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Remove this checkpoint (it's been consumed)
    shutil.rmtree(str(cp_dir))

    logger.info(f"Restored checkpoint {checkpoint_id}")

    return {
        "checkpoint_id": checkpoint_id,
        "label": meta.get("label", ""),
        "remaining": len(list_checkpoints(project_dir)),
    }


def list_checkpoints(project_dir: Path) -> list[dict]:
    """Return available checkpoints, newest first."""
    undo_dir = project_dir / "undo"
    if not undo_dir.exists():
        return []

    result = []
    for cp_dir in sorted(undo_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not cp_dir.is_dir():
            continue
        meta_path = cp_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        result.append({
            "id": cp_dir.name,
            "label": meta.get("label", ""),
            "timestamp": meta.get("timestamp", 0),
        })

    return result


def _prune_checkpoints(undo_dir: Path):
    """Keep only the most recent MAX_CHECKPOINTS checkpoints."""
    checkpoints = sorted(
        [c for c in undo_dir.iterdir() if c.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for old_cp in checkpoints[MAX_CHECKPOINTS:]:
        shutil.rmtree(str(old_cp))
        logger.info(f"Pruned old checkpoint {old_cp.name}")
