#!/usr/bin/env python3
"""Compare feed-forward 3DGS models on phone video.

Runs AnySplat, SPFSplat, NoPoSplat at different sampling rates on a video,
creates web viewer projects for each, and computes no-reference metrics.

Usage:
    python -m scripts.run_feedforward_comparison --video /path/to/video.mp4
    python -m scripts.run_feedforward_comparison --video /path/to/video.mp4 --models anysplat spfsplat
"""

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.evaluate_noreference import compute_gaussian_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ANYSPLAT_DIR = Path("/home/nock/projects/anysplat")
NOPOSPLAT_DIR = Path("/home/nock/projects/noposplat")
SPFSPLAT_DIR = Path("/home/nock/projects/spfsplat")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "feedforward_comparison"


def extract_frames(video_path: Path, output_dir: Path, n_frames: int) -> int:
    """Extract evenly-spaced frames from video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*.jpg"):
        f.unlink()

    ffmpeg = os.path.expanduser("~/.local/bin/ffmpeg")
    if not Path(ffmpeg).exists():
        ffmpeg = "ffmpeg"

    # Get total frame count
    probe = subprocess.run(
        [ffmpeg, "-i", str(video_path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    total_frames = 300
    for line in probe.stderr.split("\n"):
        m = re.search(r"frame=\s*(\d+)", line)
        if m:
            total_frames = int(m.group(1))

    step = max(1, total_frames // n_frames)
    subprocess.run(
        [
            ffmpeg, "-i", str(video_path),
            "-vf", f"select=not(mod(n\\,{step}))",
            "-vsync", "vfr", "-frames:v", str(n_frames),
            "-q:v", "2",
            str(output_dir / "frame_%05d.jpg"),
        ],
        capture_output=True,
    )

    count = len(list(output_dir.glob("frame_*.jpg")))
    logger.info(f"Extracted {count} frames (step={step}) from {video_path.name}")
    return count


def run_anysplat(frames_dir: Path, output_dir: Path, n_views: int) -> tuple[Path, dict]:
    """Run AnySplat via subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "scene.ply"
    meta_path = output_dir / "meta.json"

    python = str(ANYSPLAT_DIR / ".venv" / "bin" / "python3")
    script = str(ANYSPLAT_DIR / "run_inference.py")

    env = os.environ.copy()
    env["CUDA_HOME"] = os.path.expanduser("~/.local/cuda-12.8")

    result = subprocess.run(
        [python, script,
         "--images_dir", str(frames_dir),
         "--output_ply", str(ply_path),
         "--max_views", str(n_views),
         "--output_meta", str(meta_path)],
        capture_output=True, text=True, env=env, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"AnySplat failed:\n{result.stderr}")

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info(f"AnySplat: {meta['n_gaussians']:,} Gaussians in {meta['elapsed_sec']}s")
    return ply_path, meta


def run_spfsplat(frames_dir: Path, output_dir: Path, n_views: int) -> tuple[Path, dict]:
    """Run SPFSplat via standalone inference script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "scene.ply"
    meta_path = output_dir / "meta.json"

    python = str(SPFSPLAT_DIR / ".venv" / "bin" / "python3")
    script = str(SPFSPLAT_DIR / "run_inference.py")

    env = os.environ.copy()
    env["CUDA_HOME"] = os.path.expanduser("~/.local/cuda-12.8")

    result = subprocess.run(
        [python, script,
         "--images_dir", str(frames_dir),
         "--output_ply", str(ply_path),
         "--max_views", str(n_views),
         "--output_meta", str(meta_path)],
        capture_output=True, text=True, env=env, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"SPFSplat failed:\n{result.stderr[-2000:]}")

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info(f"SPFSplat: {meta['n_gaussians']:,} Gaussians in {meta['elapsed_sec']}s")
    return ply_path, meta


def run_noposplat(frames_dir: Path, output_dir: Path, n_views: int = 3) -> tuple[Path, dict]:
    """Run NoPoSplat via standalone inference script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "scene.ply"
    meta_path = output_dir / "meta.json"

    python = str(NOPOSPLAT_DIR / ".venv" / "bin" / "python3")
    script = str(NOPOSPLAT_DIR / "run_inference.py")
    max_views = min(n_views, 3)

    env = os.environ.copy()
    env["CUDA_HOME"] = os.path.expanduser("~/.local/cuda-12.8")

    result = subprocess.run(
        [python, script,
         "--images_dir", str(frames_dir),
         "--output_ply", str(ply_path),
         "--max_views", str(max_views),
         "--output_meta", str(meta_path)],
        capture_output=True, text=True, env=env, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"NoPoSplat failed:\n{result.stderr[-2000:]}")

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info(f"NoPoSplat: {meta['n_gaussians']:,} Gaussians in {meta['elapsed_sec']}s")
    return ply_path, meta


def create_project(name: str, ply_path: Path, meta: dict) -> str:
    """Create a project entry in the web viewer database."""
    project_id = str(uuid.uuid4())
    project_dir = DATA_DIR / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ply_path, project_dir / "scene.ply")

    db_path = Path(__file__).parent.parent / "roomeditor.db"
    conn = sqlite3.connect(str(db_path))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        "INSERT INTO projects (id, name, status, gaussian_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, name, "ready", meta.get("n_gaussians", 0), now, now),
    )
    conn.commit()
    conn.close()

    logger.info(f"Created project '{name}' ({project_id})")
    return project_id


def main():
    parser = argparse.ArgumentParser(description="Compare feed-forward 3DGS models")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["anysplat", "spfsplat", "noposplat"],
                        choices=["anysplat", "spfsplat", "noposplat"])
    parser.add_argument("--views", nargs="+", type=int, default=[3, 8, 16])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.video.exists():
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)

    run_dir = args.results_dir / args.video.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # Extract max frames needed
    frames_dir = run_dir / "frames"
    extract_frames(args.video, frames_dir, max(args.views) * 2)

    results = []
    model_runners = {
        "anysplat": run_anysplat,
        "spfsplat": run_spfsplat,
        "noposplat": run_noposplat,
    }

    for model_name in args.models:
        for n_views in args.views:
            if model_name == "noposplat" and n_views > 3:
                continue

            run_name = f"{model_name}_{n_views}v"
            run_output = run_dir / run_name
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running {run_name}")
            logger.info(f"{'=' * 60}")

            try:
                ply_path, meta = model_runners[model_name](frames_dir, run_output, n_views)

                # Gaussian stats
                gs_stats = compute_gaussian_stats(ply_path)
                gs_dict = {
                    "n_gaussians": gs_stats.n_gaussians,
                    "n_effective": gs_stats.n_effective,
                    "frac_transparent": round(gs_stats.frac_transparent, 4),
                    "frac_opaque": round(gs_stats.frac_opaque, 4),
                    "opacity_mean": round(gs_stats.opacity_mean, 4),
                    "log_scale_mean": round(gs_stats.log_scale_mean, 4),
                    "log_scale_std": round(gs_stats.log_scale_std, 4),
                    "frac_scale_outlier": round(gs_stats.frac_scale_outlier, 4),
                    "density": round(gs_stats.density, 2),
                }

                # Create web viewer project
                proj_name = f"{args.video.stem} - {model_name.upper()} ({n_views}v)"
                project_id = create_project(proj_name, ply_path, meta)

                result = {"run_name": run_name, **meta, "gaussian_stats": gs_dict, "project_id": project_id}
                results.append(result)

                with open(run_output / "result.json", "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                logger.error(f"FAILED {run_name}: {e}")
                results.append({"run_name": run_name, "error": str(e)})

    # Save combined results
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 100}")
    print(f"Feed-Forward Comparison: {args.video.name}")
    print(f"{'=' * 100}")
    print(f"{'Method':<22} {'Views':>5} {'#GS':>10} {'Eff.':>10} {'Time':>7} {'VRAM':>6} {'OpaMu':>6} {'Trans%':>7} {'Opq%':>6} {'ScStd':>6}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['run_name']:<22} ERROR: {r['error'][:60]}")
            continue
        gs = r.get("gaussian_stats", {})
        print(
            f"{r['run_name']:<22} "
            f"{r.get('n_views', '?'):>5} "
            f"{r.get('n_gaussians', 0):>10,} "
            f"{gs.get('n_effective', 0):>10,} "
            f"{r.get('elapsed_sec', 0):>6.1f}s "
            f"{r.get('vram_peak_gb', 0):>5.1f}G "
            f"{gs.get('opacity_mean', 0):>6.3f} "
            f"{gs.get('frac_transparent', 0):>6.1%} "
            f"{gs.get('frac_opaque', 0):>5.1%} "
            f"{gs.get('log_scale_std', 0):>6.2f}"
        )


if __name__ == "__main__":
    main()
