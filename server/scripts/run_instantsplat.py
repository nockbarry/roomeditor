#!/usr/bin/env python3
"""Run InstantSplat on benchmark scenes via subprocess.

Does NOT import InstantSplat code — avoids gsplat/diff-gaussian-rasterization conflict.
Runs InstantSplat's scripts using its own venv Python, then collects results.

Usage:
    python scripts/run_instantsplat.py --scene kitchen --dataset mipnerf360
    python scripts/run_instantsplat.py --all-indoor --dataset mipnerf360
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.benchmark import (
    DEFAULT_BENCHMARKS_DIR,
    MIPNERF360_ALL,
    MIPNERF360_INDOOR,
    MIPNERF360_OUTDOOR,
    get_scene,
)
from pipeline.evaluate import split_train_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

INSTANTSPLAT_DIR = Path("/home/nock/projects/instantsplat")
INSTANTSPLAT_PYTHON = INSTANTSPLAT_DIR / ".venv" / "bin" / "python"
DEFAULT_RESULTS_DIR = (
    Path(__file__).parent.parent / "data" / "benchmarks" / "results"
)


def check_instantsplat():
    """Verify InstantSplat is set up."""
    if not INSTANTSPLAT_DIR.exists():
        logger.error(
            f"InstantSplat not found at {INSTANTSPLAT_DIR}\n"
            f"Run: bash scripts/setup_instantsplat.sh"
        )
        sys.exit(1)
    if not INSTANTSPLAT_PYTHON.exists():
        logger.error(
            f"InstantSplat venv not found at {INSTANTSPLAT_PYTHON}\n"
            f"Run: bash scripts/setup_instantsplat.sh"
        )
        sys.exit(1)


def run_scene(
    scene_name: str,
    dataset: str,
    benchmarks_dir: Path,
    results_dir: Path,
    downscale: int = 4,
    test_every: int = 8,
    n_views: int = 0,
) -> dict | None:
    """Run InstantSplat on a single benchmark scene.

    Args:
        scene_name: Scene name (e.g., "kitchen").
        dataset: Dataset name (e.g., "mipnerf360").
        benchmarks_dir: Root benchmarks dir.
        results_dir: Where to store results.
        downscale: Image downscale factor.
        test_every: Test view interval.
        n_views: Number of input views (0 = all train views).

    Returns:
        Result dict or None on failure.
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"InstantSplat: {dataset}/{scene_name}")
    logger.info(f"{'=' * 60}")

    scene = get_scene(benchmarks_dir, dataset, scene_name, downscale=downscale)

    # Count total images
    image_files = sorted(scene.images_dir.glob("*.jpg")) + sorted(
        scene.images_dir.glob("*.JPG")
    ) + sorted(scene.images_dir.glob("*.png"))
    n_total = len(image_files)

    train_indices, test_indices = split_train_test(n_total, test_every)
    logger.info(f"Total images: {n_total}, Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Set up working directory for InstantSplat
    output_dir = results_dir / "instantsplat" / dataset / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "workspace"
    work_dir.mkdir(exist_ok=True)

    # Symlink images into InstantSplat's expected structure
    input_dir = work_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()

    # Use only train images as input
    train_images = [image_files[i] for i in train_indices]
    if n_views > 0:
        # Subsample evenly
        step = max(1, len(train_images) // n_views)
        train_images = train_images[::step][:n_views]

    for img in train_images:
        (input_dir / img.name).symlink_to(img)

    logger.info(f"Using {len(train_images)} input images")

    # Run InstantSplat — the exact command depends on their repo structure.
    # We try the standard entry point.
    t_start = time.time()

    env = os.environ.copy()
    env["CUDA_HOME"] = os.path.expanduser("~/.local/cuda-12.8")
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # InstantSplat typically has a run script — try common patterns
    run_script = INSTANTSPLAT_DIR / "run.py"
    if not run_script.exists():
        run_script = INSTANTSPLAT_DIR / "train.py"
    if not run_script.exists():
        # Look for the coarse-to-fine pipeline
        run_script = INSTANTSPLAT_DIR / "coarse_train_eval.py"

    if not run_script.exists():
        logger.error(
            f"Could not find InstantSplat entry point. "
            f"Checked: run.py, train.py, coarse_train_eval.py"
        )
        return None

    cmd = [
        str(INSTANTSPLAT_PYTHON),
        str(run_script),
        "--source_path", str(work_dir),
        "--model_path", str(output_dir / "model"),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(INSTANTSPLAT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        training_time = time.time() - t_start

        if proc.returncode != 0:
            logger.error(f"InstantSplat failed (exit {proc.returncode})")
            logger.error(f"stdout: {proc.stdout[-2000:]}")
            logger.error(f"stderr: {proc.stderr[-2000:]}")
            return None

        logger.info(f"InstantSplat completed in {training_time:.1f}s")

    except subprocess.TimeoutExpired:
        logger.error("InstantSplat timed out after 1 hour")
        return None

    # Collect results — look for standard output files
    # InstantSplat typically outputs results.json or metrics in model dir
    model_dir = output_dir / "model"
    metrics = _collect_metrics(model_dir)

    result = {
        "method": "instantsplat",
        "dataset": dataset,
        "scene": scene_name,
        "config": {
            "n_input_views": len(train_images),
            "downscale": downscale,
        },
        "training_time_sec": round(training_time, 1),
        "num_gaussians": metrics.get("num_gaussians", 0),
        "metrics": {
            "mean_psnr": metrics.get("psnr", 0.0),
            "mean_ssim": metrics.get("ssim", 0.0),
            "mean_lpips": metrics.get("lpips", 0.0),
        },
        "per_view": metrics.get("per_view", []),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return result


def _collect_metrics(model_dir: Path) -> dict:
    """Try to collect metrics from InstantSplat output."""
    metrics = {}

    # Check for standard results files
    for candidate in [
        model_dir / "results.json",
        model_dir / "eval" / "results.json",
        model_dir / "test" / "results.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            # Try to extract metrics from various formats
            if "PSNR" in data:
                metrics["psnr"] = data["PSNR"]
            elif "psnr" in data:
                metrics["psnr"] = data["psnr"]
            if "SSIM" in data:
                metrics["ssim"] = data["SSIM"]
            elif "ssim" in data:
                metrics["ssim"] = data["ssim"]
            if "LPIPS" in data:
                metrics["lpips"] = data["LPIPS"]
            elif "lpips" in data:
                metrics["lpips"] = data["lpips"]
            break

    # Count Gaussians from PLY if available
    for ply in model_dir.glob("**/*.ply"):
        try:
            with open(ply, "rb") as f:
                header = b""
                while True:
                    line = f.readline()
                    header += line
                    if b"end_header" in line:
                        break
                    if line.startswith(b"element vertex"):
                        metrics["num_gaussians"] = int(
                            line.split()[-1]
                        )
        except Exception:
            pass
        break

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run InstantSplat on benchmark scenes"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", type=str, help="Single scene name")
    group.add_argument("--all-indoor", action="store_true")
    group.add_argument("--all-outdoor", action="store_true")
    group.add_argument("--all", action="store_true")

    parser.add_argument("--dataset", default="mipnerf360")
    parser.add_argument("--downscale", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--n-views", type=int, default=0,
                        help="Number of input views (0=all train views)")
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args()

    check_instantsplat()

    if args.scene:
        scenes = [args.scene]
    elif args.all_indoor:
        scenes = MIPNERF360_INDOOR
    elif args.all_outdoor:
        scenes = MIPNERF360_OUTDOOR
    else:
        scenes = MIPNERF360_ALL

    results = []
    for scene_name in scenes:
        result = run_scene(
            scene_name=scene_name,
            dataset=args.dataset,
            benchmarks_dir=args.benchmarks_dir,
            results_dir=args.results_dir,
            downscale=args.downscale,
            test_every=args.test_every,
            n_views=args.n_views,
        )
        if result is not None:
            results.append(result)

    if results:
        print(f"\n{'=' * 60}")
        print("InstantSplat Results")
        print(f"{'=' * 60}")
        print(f"{'Scene':<15} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Time':>8}")
        print("-" * 55)
        for r in results:
            m = r["metrics"]
            t = r["training_time_sec"]
            mins = int(t // 60)
            secs = int(t % 60)
            print(
                f"{r['scene']:<15} {m['mean_psnr']:>8.2f} "
                f"{m['mean_ssim']:>8.4f} {m['mean_lpips']:>8.4f} "
                f"{mins:>4}m{secs:02d}s"
            )


if __name__ == "__main__":
    main()
