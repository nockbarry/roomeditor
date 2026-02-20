#!/usr/bin/env python3
"""Validate the 2DGS trainer on Mip-NeRF 360 benchmark with ground-truth COLMAP poses.

This script uses the pre-computed COLMAP poses from the Mip-NeRF 360 dataset
(not our own COLMAP run) to validate that the Gaussian training pipeline itself
works correctly. If we can't hit ~28+ PSNR on Kitchen with ground-truth poses,
the training code has bugs. If we can, the problem is confirmed as input quality.

Usage:
    # First download the benchmark data:
    python scripts/download_benchmarks.py --dataset mipnerf360

    # Then run validation:
    python scripts/run_benchmark_validation.py --scene kitchen
    python scripts/run_benchmark_validation.py --scene kitchen --iterations 30000 --sh-degree 3
    python scripts/run_benchmark_validation.py --all-indoor

    # Compare MASt3R vs COLMAP on same data:
    python scripts/run_benchmark_validation.py --scene kitchen --sfm-backend mast3r
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.benchmark import (
    DEFAULT_BENCHMARKS_DIR,
    MIPNERF360_ALL,
    MIPNERF360_INDOOR,
    MIPNERF360_OUTDOOR,
    get_scene,
)
from pipeline.evaluate import EvalResults, split_train_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "results"


def run_validation(
    scene_name: str,
    dataset: str,
    benchmarks_dir: Path,
    results_dir: Path,
    iterations: int = 30_000,
    sh_degree: int = 3,
    mode: str = "2dgs",
    downscale: int = 4,
    test_every: int = 8,
    sfm_backend: str = "gt",
    mast3r_image_size: int = 512,
) -> dict | None:
    """Run 2DGS training on a benchmark scene and evaluate.

    When sfm_backend="gt", uses the pre-computed COLMAP poses from the dataset.
    When sfm_backend="mast3r", runs MASt3R for pose estimation.
    """
    import torch

    logger.info(f"{'=' * 60}")
    logger.info(f"Benchmark validation: {dataset}/{scene_name}")
    logger.info(f"  Mode: {mode}, Iter: {iterations}, SH: {sh_degree}, SfM: {sfm_backend}")
    logger.info(f"{'=' * 60}")

    scene = get_scene(benchmarks_dir, dataset, scene_name, downscale=downscale)

    # Count images
    image_files = sorted(scene.images_dir.glob("*.jpg")) + \
                  sorted(scene.images_dir.glob("*.JPG")) + \
                  sorted(scene.images_dir.glob("*.png"))
    n_total = len(image_files)
    train_indices, test_indices = split_train_test(n_total, test_every)

    logger.info(f"Total images: {n_total}, Train: {len(train_indices)}, Test: {len(test_indices)}")

    # Set up output directory
    method_name = f"validation-{mode}-{sfm_backend}"
    output_dir = results_dir / method_name / dataset / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which COLMAP model to use
    if sfm_backend == "mast3r":
        # Run MASt3R on train images
        from pipeline.run_mast3r import run_mast3r

        mast3r_dir = output_dir / "mast3r"
        model_dir, sfm_meta = run_mast3r(
            images_dir=scene.images_dir,
            output_dir=mast3r_dir,
            image_size=mast3r_image_size,
            progress_callback=lambda f: logger.info(f"  MASt3R: {f*100:.0f}%") if f in (0.5, 1.0) else None,
        )
        logger.info(f"MASt3R: {sfm_meta}")
    else:
        # Use ground-truth COLMAP poses from the dataset
        model_dir = scene.colmap_dir

    # Train Gaussians
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig, TrainerCallbacks

    config = TrainerConfig(
        iterations=iterations,
        sh_degree=sh_degree,
        mode=mode,
        depth_reg_weight=0.0,
        distortion_weight=0.0,
        normal_weight=0.05 if mode == "2dgs" else 0.0,
        scale_reg_weight=0.0,
        prune_opa=0.005,
        densify_until_pct=0.5,
    )

    trainer = GaussianTrainer(config)
    t_start = time.time()

    logger.info("Loading data...")
    points3d, colors3d = trainer.load_data(model_dir, scene.images_dir)

    # Split into train/test
    all_cameras = list(trainer.cameras_data)
    all_images = list(trainer.gt_images)

    # Adjust train_indices and test_indices if we have fewer cameras than total images
    actual_n = len(all_cameras)
    adj_train = [i for i in train_indices if i < actual_n]
    adj_test = [i for i in test_indices if i < actual_n]

    trainer.set_training_views(adj_train)

    trainer.init_params(points3d, colors3d)
    trainer.init_strategy()

    logger.info(f"Training {iterations} iterations...")
    ply_path = output_dir / "model.ply"

    def _progress(frac):
        if frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            logger.info(f"  Training: {frac*100:.0f}%")

    callbacks = TrainerCallbacks(progress=_progress)
    n_gaussians = trainer.train(callbacks, output_path=ply_path)
    trainer.export(ply_path)

    training_time = time.time() - t_start
    logger.info(f"Training complete in {training_time:.1f}s: {n_gaussians} Gaussians")

    # Evaluate on test views
    logger.info("Evaluating on test views...")
    from pipeline.evaluate import evaluate_model

    renders_dir = output_dir / "renders"
    eval_results = evaluate_model(
        trainer, adj_test, all_cameras, all_images, save_dir=renders_dir
    )

    trainer.cleanup()

    eval_results.print_table()

    # Save results
    result = {
        "method": method_name,
        "dataset": dataset,
        "scene": scene_name,
        "config": {
            "iterations": iterations,
            "sh_degree": sh_degree,
            "mode": mode,
            "sfm_backend": sfm_backend,
            "downscale": downscale,
        },
        "training_time_sec": round(training_time, 1),
        "num_gaussians": n_gaussians,
        "metrics": {
            "mean_psnr": eval_results.mean_psnr,
            "mean_ssim": eval_results.mean_ssim,
            "mean_lpips": eval_results.mean_lpips,
        },
        "per_view": eval_results.per_view,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate 2DGS trainer on benchmarks")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", type=str, help="Single scene name (e.g., kitchen)")
    group.add_argument("--all-indoor", action="store_true")
    group.add_argument("--all-outdoor", action="store_true")
    group.add_argument("--all", action="store_true")

    parser.add_argument("--dataset", default="mipnerf360")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--mode", default="2dgs", choices=["2dgs", "3dgs"])
    parser.add_argument("--downscale", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--sfm-backend", default="gt", choices=["gt", "mast3r"],
                        help="gt=use dataset COLMAP poses, mast3r=run MASt3R")
    parser.add_argument("--mast3r-image-size", type=int, default=512)
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args()

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
        result = run_validation(
            scene_name=scene_name,
            dataset=args.dataset,
            benchmarks_dir=args.benchmarks_dir,
            results_dir=args.results_dir,
            iterations=args.iterations,
            sh_degree=args.sh_degree,
            mode=args.mode,
            downscale=args.downscale,
            test_every=args.test_every,
            sfm_backend=args.sfm_backend,
            mast3r_image_size=args.mast3r_image_size,
        )
        if result is not None:
            results.append(result)

    if results:
        print(f"\n{'=' * 70}")
        print(f"Benchmark Validation Results ({args.mode}, sfm={args.sfm_backend})")
        print(f"{'=' * 70}")
        print(f"{'Scene':<15} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Time':>10} {'#GS':>10}")
        print("-" * 65)
        for r in results:
            m = r["metrics"]
            t = r["training_time_sec"]
            mins = int(t // 60)
            secs = int(t % 60)
            print(
                f"{r['scene']:<15} {m['mean_psnr']:>8.2f} "
                f"{m['mean_ssim']:>8.4f} {m['mean_lpips']:>8.4f} "
                f"{mins:>4}m{secs:02d}s {r['num_gaussians']:>10,}"
            )

        # Published baselines for comparison
        print(f"\n{'Published Baselines (Mip-NeRF 360 Indoor)':}")
        print(f"  3DGS:          ~31 PSNR / 0.92 SSIM")
        print(f"  2DGS:          ~30 PSNR / 0.90 SSIM")
        print(f"  Mip-Splatting: ~31 PSNR / 0.92 SSIM")


if __name__ == "__main__":
    main()
