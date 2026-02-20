#!/usr/bin/env python3
"""Run InstantSplat on the kitchen video project.

Sets up the required directory structure, runs InstantSplat's 3-step pipeline
(MASt3R init → 3DGS training → render + eval), then saves results to the
benchmarks directory and creates a viewer project.

Runs InstantSplat via subprocess using its own venv to avoid dependency conflicts.

Usage:
    cd server
    CUDA_HOME=~/.local/cuda-12.8 python scripts/run_instantsplat_kitchen.py
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.evaluate import split_train_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
SERVER_DIR = Path(__file__).parent.parent
DATA_DIR = SERVER_DIR / "data"
SOURCE_PROJECT = "d7a93e8d-a6fa-46eb-b1da-199aba06559a"
PROJECT_DIR = DATA_DIR / "projects" / SOURCE_PROJECT
FRAMES_DIR = PROJECT_DIR / "frames"

INSTANTSPLAT_DIR = Path("/home/nock/projects/instantsplat")
INSTANTSPLAT_PYTHON = INSTANTSPLAT_DIR / ".venv" / "bin" / "python"
CKPT_PATH = INSTANTSPLAT_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

RESULTS_DIR = DATA_DIR / "benchmarks" / "results"
DATASET = "kitchen-video"
SCENE = "kitchen"
TEST_EVERY = 8

# InstantSplat settings
N_VIEWS = 12  # Number of views for MASt3R init (all-pairs is O(n^2), 24 OOMs on 16GB)
GS_TRAIN_ITER = 2000  # InstantSplat default; short because init is strong


def check_setup():
    """Verify InstantSplat is installed."""
    if not INSTANTSPLAT_DIR.exists():
        logger.error(f"InstantSplat not found at {INSTANTSPLAT_DIR}")
        logger.error("Run: bash scripts/setup_instantsplat.sh")
        sys.exit(1)
    if not INSTANTSPLAT_PYTHON.exists():
        logger.error(f"InstantSplat venv not found")
        logger.error("Run: bash scripts/setup_instantsplat.sh")
        sys.exit(1)
    if not CKPT_PATH.exists():
        # Also check the mast3r/checkpoints location
        alt = INSTANTSPLAT_DIR / "mast3r" / "checkpoints" / CKPT_PATH.name
        if alt.exists():
            logger.info(f"Checkpoint found at {alt}")
            return str(alt)
        logger.error(f"MASt3R checkpoint not found at {CKPT_PATH}")
        logger.error("Run: bash scripts/setup_instantsplat.sh")
        sys.exit(1)
    return str(CKPT_PATH)


def run_subprocess(cmd, desc, log_path=None):
    """Run a subprocess with logging."""
    env = os.environ.copy()
    env["CUDA_HOME"] = os.path.expanduser("~/.local/cuda-12.8")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Mesa GL libs for WSL (no system libGL)
    mesa_lib = os.path.expanduser("~/.local/lib/mesa")
    env["LD_LIBRARY_PATH"] = mesa_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env["MPLBACKEND"] = "Agg"  # No tkinter in WSL

    logger.info(f"[{desc}] Running: {' '.join(cmd)}")
    t_start = time.time()

    proc = subprocess.run(
        cmd,
        cwd=str(INSTANTSPLAT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour max
    )

    elapsed = time.time() - t_start
    logger.info(f"[{desc}] Completed in {elapsed:.1f}s (exit code {proc.returncode})")

    if log_path:
        log_path.write_text(f"=== STDOUT ===\n{proc.stdout}\n=== STDERR ===\n{proc.stderr}\n")

    if proc.returncode != 0:
        logger.error(f"[{desc}] FAILED")
        logger.error(f"Last stdout: {proc.stdout[-3000:]}")
        logger.error(f"Last stderr: {proc.stderr[-3000:]}")
        sys.exit(1)

    return elapsed


def setup_workspace(work_dir: Path):
    """Create InstantSplat-compatible directory structure from kitchen frames."""
    images_dir = work_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    # Symlink all frames as the image source
    frames = sorted(FRAMES_DIR.glob("*.png")) + sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        logger.error(f"No frames found in {FRAMES_DIR}")
        sys.exit(1)

    for f in frames:
        (images_dir / f.name).symlink_to(f)

    logger.info(f"Linked {len(frames)} frames into {images_dir}")
    return len(frames)


def collect_ply(model_dir: Path) -> Path | None:
    """Find the output PLY from InstantSplat training."""
    # InstantSplat saves point_cloud/iteration_XXXX/point_cloud.ply
    candidates = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if candidates:
        return candidates[-1]  # Latest iteration
    return None


def count_ply_vertices(ply_path: Path) -> int:
    """Read vertex count from PLY header."""
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if b"end_header" in line:
                break
    return 0


def run_eval_with_our_pipeline(model_dir: Path, work_dir: Path, n_total: int):
    """Run our eval pipeline on InstantSplat renders for consistent metrics."""
    # InstantSplat's metrics.py outputs to model_dir/results.json
    metrics_file = model_dir / "results.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
        return data

    # Fall back to checking their per-scene results format
    for candidate in [
        model_dir / "test" / f"ours_{GS_TRAIN_ITER}" / "results.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)

    return None


def main():
    ckpt_path = check_setup()

    # Output dirs
    output_dir = RESULTS_DIR / "instantsplat" / DATASET / SCENE
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "workspace"
    model_dir = output_dir / "model"

    # Setup workspace
    n_total = setup_workspace(work_dir)
    train_indices, test_indices = split_train_test(n_total, TEST_EVERY)
    logger.info(f"Total: {n_total}, Train: {len(train_indices)}, Test: {len(test_indices)}")

    total_time = 0.0

    # Step 1: MASt3R geometry initialization
    logger.info("=" * 60)
    logger.info("Step 1: MASt3R Geometry Initialization")
    logger.info("=" * 60)
    elapsed = run_subprocess(
        [
            str(INSTANTSPLAT_PYTHON), "-W", "ignore", "./init_geo.py",
            "-s", str(work_dir),
            "-m", str(model_dir),
            "--ckpt_path", ckpt_path,
            "--n_views", str(N_VIEWS),
            "--focal_avg",
            "--co_vis_dsp",
            "--conf_aware_ranking",
            "--llffhold", str(TEST_EVERY),
        ],
        desc="init_geo",
        log_path=output_dir / "01_init_geo.log",
    )
    total_time += elapsed

    # Step 2: 3DGS Training with pose optimization
    logger.info("=" * 60)
    logger.info("Step 2: 3DGS Training")
    logger.info("=" * 60)
    elapsed = run_subprocess(
        [
            str(INSTANTSPLAT_PYTHON), "./train.py",
            "-s", str(work_dir),
            "-m", str(model_dir),
            "-r", "1",
            "--n_views", str(N_VIEWS),
            "--iterations", str(GS_TRAIN_ITER),
            "--pp_optimizer",
            "--optim_pose",
        ],
        desc="train",
        log_path=output_dir / "02_train.log",
    )
    total_time += elapsed

    # Step 3: Render test views
    logger.info("=" * 60)
    logger.info("Step 3: Render Test Views")
    logger.info("=" * 60)
    elapsed = run_subprocess(
        [
            str(INSTANTSPLAT_PYTHON), "./render.py",
            "-s", str(work_dir),
            "-m", str(model_dir),
            "-r", "1",
            "--n_views", str(N_VIEWS),
            "--iterations", str(GS_TRAIN_ITER),
            "--eval",
        ],
        desc="render",
        log_path=output_dir / "03_render.log",
    )
    total_time += elapsed

    # Step 4: Metrics
    logger.info("=" * 60)
    logger.info("Step 4: Compute Metrics")
    logger.info("=" * 60)
    elapsed = run_subprocess(
        [
            str(INSTANTSPLAT_PYTHON), "./metrics.py",
            "-s", str(work_dir),
            "-m", str(model_dir),
            "--n_views", str(N_VIEWS),
        ],
        desc="metrics",
        log_path=output_dir / "04_metrics.log",
    )
    total_time += elapsed

    # Collect results
    logger.info("=" * 60)
    logger.info("Collecting results...")
    logger.info("=" * 60)

    ply_path = collect_ply(model_dir)
    n_gaussians = count_ply_vertices(ply_path) if ply_path else 0
    logger.info(f"PLY: {ply_path}, Gaussians: {n_gaussians}")

    # Try to read InstantSplat's metrics output
    metrics = {"mean_psnr": 0.0, "mean_ssim": 0.0, "mean_lpips": 0.0}
    per_view = []

    # InstantSplat metrics.py writes results to model_dir/results.json
    for results_candidate in [
        model_dir / "results.json",
        model_dir / f"test/ours_{GS_TRAIN_ITER}/results.json",
    ]:
        if results_candidate.exists():
            logger.info(f"Found metrics at {results_candidate}")
            with open(results_candidate) as f:
                raw = json.load(f)
            # InstantSplat format: {"ours_XXXX": {"SSIM": ..., "PSNR": ..., "LPIPS": ...}}
            if isinstance(raw, dict):
                for key, vals in raw.items():
                    if isinstance(vals, dict):
                        metrics["mean_psnr"] = vals.get("PSNR", vals.get("psnr", 0.0))
                        metrics["mean_ssim"] = vals.get("SSIM", vals.get("ssim", 0.0))
                        metrics["mean_lpips"] = vals.get("LPIPS", vals.get("lpips", 0.0))
                        break
            break

    # Copy renders to standard location
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    is_render_src = model_dir / f"test/ours_{GS_TRAIN_ITER}/renders"
    if is_render_src.is_dir():
        for img in sorted(is_render_src.iterdir()):
            if img.suffix.lower() in (".png", ".jpg"):
                shutil.copy2(img, renders_dir / img.name)
        logger.info(f"Copied {len(list(renders_dir.iterdir()))} render images")

    # Save results
    result = {
        "method": "instantsplat",
        "dataset": DATASET,
        "scene": SCENE,
        "config": {
            "n_views": N_VIEWS,
            "gs_train_iter": GS_TRAIN_ITER,
            "test_every": TEST_EVERY,
            "image_size": 512,
        },
        "training_time_sec": round(total_time, 1),
        "num_gaussians": n_gaussians,
        "metrics": metrics,
        "per_view": per_view,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"InstantSplat Results:")
    logger.info(f"  PSNR:  {metrics['mean_psnr']:.2f}")
    logger.info(f"  SSIM:  {metrics['mean_ssim']:.4f}")
    logger.info(f"  LPIPS: {metrics['mean_lpips']:.4f}")
    logger.info(f"  Time:  {total_time:.1f}s")
    logger.info(f"  #GS:   {n_gaussians}")
    logger.info(f"{'=' * 60}")

    # Create a viewer project
    if ply_path and ply_path.exists():
        new_id = str(uuid.uuid4())
        new_project_dir = DATA_DIR / "projects" / new_id
        new_project_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ply_path, new_project_dir / "scene.ply")

        db_path = SERVER_DIR / "roomeditor.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO projects (id, name, status, gaussian_count, created_at, updated_at) "
            "VALUES (?, ?, 'ready', ?, datetime('now'), datetime('now'))",
            (new_id, "Kitchen InstantSplat", n_gaussians),
        )
        conn.commit()
        conn.close()

        logger.info(f"Created project {new_id}: Kitchen InstantSplat")
        logger.info(f"View at: http://localhost:5173/project/{new_id}")
    else:
        logger.warning("No PLY found — skipping project creation")

    logger.info("Done!")


if __name__ == "__main__":
    main()
