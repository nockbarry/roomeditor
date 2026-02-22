"""Run SPFSplat V2 inference as a subprocess.

SPFSplat (ICCV 2025) is a pose-free feed-forward Gaussian Splatting method.
Fixed 256x256 input resolution, max 8 views, DC-only SH, no camera export.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SPFSPLAT_DIR = Path("/home/nock/projects/spfsplat")
SPFSPLAT_PYTHON = SPFSPLAT_DIR / ".venv" / "bin" / "python3"
CUDA_HOME = os.path.expanduser("~/.local/cuda-12.8")

# SPFSplat always uses 256x256 input
SPFSPLAT_RESOLUTION = 256
SPFSPLAT_MAX_VIEWS = 8


def run_spfsplat_subprocess(
    images_dir: Path,
    output_ply: Path,
    max_views: int = 8,
    progress_callback: Callable[[float], None] | None = None,
) -> int:
    """Run SPFSplat inference as a subprocess.

    SPFSplat is pose-free: no camera poses needed, no cameras.json exported.
    Fixed 256x256 resolution, DC-only SH.

    Args:
        images_dir: Directory containing input images/frames.
        output_ply: Output PLY file path.
        max_views: Maximum views (capped at 8).
        progress_callback: Optional progress callback (0-1).

    Returns:
        Number of Gaussians produced.
    """
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_ply.with_suffix(".json")

    max_views = min(max_views, SPFSPLAT_MAX_VIEWS)

    env = {**os.environ, "CUDA_HOME": CUDA_HOME, "PYTHONUNBUFFERED": "1"}

    script = SPFSPLAT_DIR / "run_inference.py"
    cmd = [
        str(SPFSPLAT_PYTHON), str(script),
        "--images_dir", str(images_dir),
        "--output_ply", str(output_ply),
        "--max_views", str(max_views),
        "--resolution", str(SPFSPLAT_RESOLUTION),
        "--output_meta", str(meta_path),
    ]

    logger.info(f"SPFSplat: {max_views} views @ {SPFSPLAT_RESOLUTION}px")

    if progress_callback:
        progress_callback(0.1)

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        logger.error(f"SPFSplat failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        raise RuntimeError(f"SPFSplat failed: {result.stderr[-500:] if result.stderr else result.stdout[-500:]}")

    if progress_callback:
        progress_callback(0.9)

    # Parse metadata
    n_gaussians = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n_gaussians = meta.get("n_gaussians", 0)
        logger.info(f"SPFSplat result: {meta}")
    else:
        for line in reversed(result.stdout.strip().split("\n")):
            try:
                meta = json.loads(line)
                n_gaussians = meta.get("n_gaussians", 0)
                break
            except json.JSONDecodeError:
                continue

    if not output_ply.exists():
        raise RuntimeError("SPFSplat did not produce an output PLY file")

    if progress_callback:
        progress_callback(1.0)

    return n_gaussians
