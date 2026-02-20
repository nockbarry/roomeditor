"""Keyframe selection: remove blurry and redundant frames from extracted video frames."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

logger = logging.getLogger(__name__)


def _laplacian_variance(img_path: Path, resize_to: int = 480) -> float:
    """Compute Laplacian variance as a sharpness metric. Higher = sharper."""
    img = Image.open(img_path).convert("L")
    w, h = img.size
    scale = resize_to / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    laplacian = img.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
        scale=1,
        offset=128,
    ))
    arr = np.array(laplacian, dtype=np.float32) - 128.0
    return float(np.var(arr))


def _frame_difference(path_a: Path, path_b: Path, resize_to: int = 160) -> float:
    """Compute mean absolute pixel difference between two frames. Range [0, 1]."""
    img_a = Image.open(path_a).convert("RGB").resize((resize_to, resize_to), Image.LANCZOS)
    img_b = Image.open(path_b).convert("RGB").resize((resize_to, resize_to), Image.LANCZOS)
    arr_a = np.array(img_a, dtype=np.float32) / 255.0
    arr_b = np.array(img_b, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(arr_a - arr_b)))


async def select_keyframes(
    frames_dir: Path,
    blur_threshold: float = 50.0,
    similarity_threshold: float = 0.02,
    min_frames: int = 20,
) -> tuple[int, int]:
    """Remove blurry and redundant frames. Returns (kept, removed).

    1. Score all frames for sharpness (Laplacian variance).
    2. Remove blurry frames (below blur_threshold), but never create gaps > 3 consecutive.
    3. Remove frames too similar to their predecessor (below similarity_threshold).
    4. Never reduce below min_frames.
    5. Renumber remaining frames sequentially.
    """
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    total = len(frames)

    if total <= min_frames:
        logger.info(f"Keyframe selection: only {total} frames, skipping filtering")
        return total, 0

    # Score sharpness for all frames
    sharpness = {}
    for f in frames:
        sharpness[f] = _laplacian_variance(f)

    keep = set(frames)

    # Pass 1: Remove blurry frames (but don't create big gaps)
    consecutive_removed = 0
    for i, f in enumerate(frames):
        if sharpness[f] < blur_threshold:
            # Don't remove if it would create a gap > 3
            if consecutive_removed < 3 and len(keep) > min_frames:
                keep.discard(f)
                consecutive_removed += 1
            else:
                consecutive_removed = 0
        else:
            consecutive_removed = 0

    # Pass 2: Remove frames too similar to predecessor
    kept_ordered = sorted(keep, key=lambda f: f.name)
    for i in range(1, len(kept_ordered)):
        if len(keep) <= min_frames:
            break
        diff = _frame_difference(kept_ordered[i - 1], kept_ordered[i])
        if diff < similarity_threshold:
            # Remove the one with lower sharpness
            if sharpness[kept_ordered[i]] < sharpness[kept_ordered[i - 1]]:
                keep.discard(kept_ordered[i])
            else:
                keep.discard(kept_ordered[i - 1])

    removed_count = total - len(keep)

    # Delete removed frames
    for f in frames:
        if f not in keep:
            f.unlink()

    # Renumber remaining frames sequentially (required for COLMAP sequential matching)
    remaining = sorted(frames_dir.glob("frame_*.jpg"))
    temp_dir = frames_dir / "_renumber_temp"
    temp_dir.mkdir(exist_ok=True)

    for idx, f in enumerate(remaining):
        new_name = f"frame_{idx:05d}.jpg"
        shutil.move(str(f), str(temp_dir / new_name))

    for f in temp_dir.iterdir():
        shutil.move(str(f), str(frames_dir / f.name))
    temp_dir.rmdir()

    kept_count = len(remaining)
    logger.info(
        f"Keyframe selection: {kept_count}/{total} frames kept "
        f"({removed_count} removed: blurry/redundant)"
    )
    return kept_count, removed_count


def _source_type_for(source_filename: str) -> str:
    """Determine source type ('video', 'image', or '') from filename extension."""
    if not source_filename:
        return ""
    ext = Path(source_filename).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return ""


async def score_and_select_frames(
    frames_dir: Path,
    blur_threshold: float = 50.0,
    similarity_threshold: float = 0.02,
    source_map: dict[str, str] | None = None,
) -> dict:
    """Non-destructive keyframe scoring. Does NOT delete files or renumber.

    Computes sharpness scores for all frames, marks blurry/redundant as
    selected: false, and writes frames/frame_manifest.json.

    Returns the manifest dict.
    """
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    total = len(frames)

    if total == 0:
        manifest = {"frames": [], "total": 0, "selected_count": 0}
        (frames_dir / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    # Score sharpness for all frames
    sharpness = {}
    for f in frames:
        sharpness[f] = _laplacian_variance(f)

    # Determine which frames to select
    selected = {f: True for f in frames}

    # Pass 1: Mark blurry frames (but don't create big gaps)
    consecutive_removed = 0
    for f in frames:
        if sharpness[f] < blur_threshold:
            if consecutive_removed < 3:
                selected[f] = False
                consecutive_removed += 1
            else:
                consecutive_removed = 0
        else:
            consecutive_removed = 0

    # Pass 2: Mark frames too similar to predecessor
    kept_ordered = [f for f in frames if selected[f]]
    for i in range(1, len(kept_ordered)):
        diff = _frame_difference(kept_ordered[i - 1], kept_ordered[i])
        if diff < similarity_threshold:
            # Mark the one with lower sharpness
            if sharpness[kept_ordered[i]] < sharpness[kept_ordered[i - 1]]:
                selected[kept_ordered[i]] = False
            else:
                selected[kept_ordered[i - 1]] = False

    # Build manifest
    frame_infos = []
    for f in frames:
        src = source_map.get(f.name, "") if source_map else ""
        frame_infos.append({
            "filename": f.name,
            "source_file": src,
            "source_type": _source_type_for(src),
            "sharpness": round(sharpness[f], 2),
            "selected": selected[f],
        })

    selected_count = sum(1 for fi in frame_infos if fi["selected"])
    manifest = {
        "frames": frame_infos,
        "total": total,
        "selected_count": selected_count,
    }

    manifest_path = frames_dir / "frame_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        f"Frame scoring: {selected_count}/{total} frames selected "
        f"({total - selected_count} marked as blurry/redundant)"
    )
    return manifest
