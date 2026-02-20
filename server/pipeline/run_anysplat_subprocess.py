"""Run AnySplat inference as a subprocess.

Uses run_inference_max.py in /home/nock/projects/anysplat/ with fp16 and
automatic resolution selection to maximize single-pass quality.

VRAM budget (16GB GPU):
  - 32 views @ 448px fp16 -> 12.5GB
  - 64 views @ 336px fp16 -> 13.8GB
  - 128 views @ 224px fp16 -> 12.6GB
"""
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

ANYSPLAT_DIR = Path("/home/nock/projects/anysplat")
ANYSPLAT_PYTHON = ANYSPLAT_DIR / ".venv" / "bin" / "python3"
CUDA_HOME = os.path.expanduser("~/.local/cuda-12.8")

# View count -> resolution mapping for 16GB VRAM budget
# Each tier stays comfortably under 16GB with fp16
RESOLUTION_TIERS = [
    (32, 448),   # up to 32 views: full resolution (12.5GB)
    (64, 336),   # 33-64 views: medium resolution (13.8GB)
    (128, 224),  # 65-128 views: lower resolution (12.6GB)
]


def pick_resolution(max_views: int) -> int:
    """Select optimal resolution for the given view count."""
    for threshold, res in RESOLUTION_TIERS:
        if max_views <= threshold:
            return res
    return 224  # fallback for >128 views


def run_anysplat_subprocess(
    images_dir: Path,
    output_ply: Path,
    max_views: int = 32,
    chunked: bool = False,
    chunk_size: int = 32,
    chunk_overlap: int = 8,
    resolution: int = 0,
    progress_callback: Callable[[float], None] | None = None,
) -> int:
    """Run AnySplat inference as a subprocess.

    Uses fp16 with automatic resolution selection for optimal quality.
    When chunked=True, processes views in overlapping chunks and merges
    the resulting PLY files, allowing unlimited views at higher resolution.

    Args:
        images_dir: Directory containing input images/frames.
        output_ply: Output PLY file path.
        max_views: Maximum number of views to use.
        chunked: If True, process in overlapping chunks and merge.
        chunk_size: Views per chunk (default 32).
        chunk_overlap: Overlap between chunks (default 8).
        resolution: Override resolution (0 = auto-select).
        progress_callback: Optional progress callback (0-1).

    Returns:
        Number of Gaussians produced.
    """
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_ply.with_suffix(".json")

    if resolution <= 0:
        resolution = pick_resolution(max_views)

    env = {**os.environ, "CUDA_HOME": CUDA_HOME, "PYTHONUNBUFFERED": "1"}

    cameras_path = output_ply.with_name("cameras.json")
    script = ANYSPLAT_DIR / "run_inference_max.py"
    cmd = [
        str(ANYSPLAT_PYTHON), str(script),
        "--images_dir", str(images_dir),
        "--output_ply", str(output_ply),
        "--max_views", str(max_views),
        "--fp16",
        "--resolution", str(resolution),
        "--output_meta", str(meta_path),
        "--output_cameras", str(cameras_path),
        "--full_sh",
    ]

    if chunked and chunk_size > 0:
        cmd.extend(["--chunk_size", str(chunk_size), "--chunk_overlap", str(chunk_overlap)])

    mode_str = f"chunked({chunk_size}/{chunk_overlap})" if chunked else "single-pass"
    logger.info(f"AnySplat: {max_views} views @ {resolution}px fp16 [{mode_str}]")

    if progress_callback:
        progress_callback(0.1)

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error(f"AnySplat failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        raise RuntimeError(f"AnySplat failed: {result.stderr[-500:] if result.stderr else result.stdout[-500:]}")

    if progress_callback:
        progress_callback(0.9)

    # Parse metadata
    n_gaussians = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n_gaussians = meta.get("n_gaussians", 0)
        logger.info(f"AnySplat result: {meta}")
    else:
        # Try to parse from last line of stdout
        for line in reversed(result.stdout.strip().split("\n")):
            try:
                meta = json.loads(line)
                n_gaussians = meta.get("n_gaussians", 0)
                break
            except json.JSONDecodeError:
                continue

    if not output_ply.exists():
        raise RuntimeError("AnySplat did not produce an output PLY file")

    if progress_callback:
        progress_callback(1.0)

    return n_gaussians
