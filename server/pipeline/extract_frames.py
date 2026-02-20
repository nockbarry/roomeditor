import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


async def extract_frames_multi(
    source_dir: Path,
    output_dir: Path,
    fps: int = 2,
    ffmpeg_path: str = "ffmpeg",
    progress_callback: callable = None,
) -> tuple[int, dict[str, str]]:
    """Extract frames from all videos and copy all photos in source_dir.

    source_dir should contain video files and/or image files.
    All frames get unified numbering in output_dir as frame_XXXXX.jpg.

    Returns (total_frames, source_map) where source_map maps frame filenames
    to their source filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing frames
    for f in output_dir.glob("*.jpg"):
        f.unlink()
    for f in output_dir.glob("*.png"):
        f.unlink()

    # Collect source files
    videos = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])
    images = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not videos and not images:
        raise RuntimeError(f"No video or image files found in {source_dir}")

    frame_idx = 0
    source_map: dict[str, str] = {}  # frame filename -> source filename
    total_sources = len(videos) + (1 if images else 0)
    completed_sources = 0

    # Extract frames from each video
    for video_path in videos:
        logger.info(f"Extracting frames from {video_path.name} at {fps}fps")
        temp_dir = output_dir / f"_temp_{video_path.stem}"
        temp_dir.mkdir(exist_ok=True)

        cmd = [
            ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            "-start_number", "0",
            str(temp_dir / "frame_%05d.jpg"),
            "-y",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            raise RuntimeError(
                f"ffmpeg failed on {video_path.name} (exit {proc.returncode}): {error_msg}"
            )

        # Move temp frames into unified output with sequential numbering
        temp_frames = sorted(temp_dir.glob("*.jpg"))
        for frame in temp_frames:
            frame_name = f"frame_{frame_idx:05d}.jpg"
            dest = output_dir / frame_name
            shutil.move(str(frame), str(dest))
            source_map[frame_name] = video_path.name
            frame_idx += 1

        shutil.rmtree(temp_dir)
        logger.info(f"  -> {len(temp_frames)} frames from {video_path.name}")
        completed_sources += 1
        if progress_callback and total_sources > 0:
            await progress_callback(completed_sources / total_sources)

    # Copy image files as additional frames
    for img_path in images:
        frame_name = f"frame_{frame_idx:05d}.jpg"
        dest = output_dir / frame_name
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(str(img_path), str(dest))
        else:
            # Convert non-JPEG images to JPEG
            cmd = [
                ffmpeg_path,
                "-i", str(img_path),
                "-q:v", "2",
                str(dest),
                "-y",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                # Fall back to just copying
                shutil.copy2(str(img_path), str(dest))
        source_map[frame_name] = img_path.name
        frame_idx += 1

    if images:
        logger.info(f"  -> {len(images)} images copied")
        completed_sources += 1
        if progress_callback and total_sources > 0:
            await progress_callback(completed_sources / total_sources)

    total = len(list(output_dir.glob("frame_*.jpg")))
    if total == 0:
        raise RuntimeError("No frames produced from any source")

    logger.info(f"Total: {total} frames in {output_dir}")
    return total, source_map
