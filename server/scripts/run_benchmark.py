#!/usr/bin/env python3
"""Run our gsplat pipeline on benchmark scenes with evaluation.

Usage:
    # Single scene
    python scripts/run_benchmark.py --scene kitchen --mode 2dgs --iterations 30000

    # All indoor scenes
    python scripts/run_benchmark.py --all-indoor --mode 2dgs --iterations 30000

    # All scenes
    python scripts/run_benchmark.py --all --mode 2dgs --iterations 30000
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from pipeline.benchmark import (
    DEFAULT_BENCHMARKS_DIR,
    MIPNERF360_ALL,
    MIPNERF360_INDOOR,
    MIPNERF360_OUTDOOR,
    get_scene,
)
from pipeline.evaluate import EvalResults, evaluate_model, split_train_test
from pipeline.train_gaussians import (
    GaussianTrainer,
    TrainerCallbacks,
    TrainerConfig,
    load_colmap_data,
    _load_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "results"


def run_scene(
    scene_name: str,
    dataset: str,
    benchmarks_dir: Path,
    results_dir: Path,
    config: TrainerConfig,
    downscale: int = 4,
    test_every: int = 8,
) -> dict:
    """Run training + evaluation on a single benchmark scene.

    Returns:
        Result dict with metrics, timing, and config.
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Scene: {dataset}/{scene_name}")
    logger.info(f"Mode: {config.mode}, Iterations: {config.iterations}")
    logger.info(f"{'=' * 60}")

    # Load scene
    scene = get_scene(benchmarks_dir, dataset, scene_name, downscale=downscale)

    # Load COLMAP data
    all_cameras, points3d, colors3d = load_colmap_data(
        scene.colmap_dir, scene.images_dir
    )

    # Load all images to GPU
    logger.info("Loading images to GPU...")
    all_images = []
    for cam in all_cameras:
        all_images.append(
            _load_image(cam["image_path"], cam["width"], cam["height"])
        )

    # Split train/test
    n_images = len(all_cameras)
    train_indices, test_indices = split_train_test(n_images, test_every)
    logger.info(
        f"Split: {len(train_indices)} train, {len(test_indices)} test "
        f"(every {test_every}th held out)"
    )

    # Create trainer
    trainer = GaussianTrainer(config)
    trainer.cameras_data = all_cameras
    trainer.gt_images = all_images

    # Downscale if images are large (trainer.load_data does this, but we
    # loaded manually to keep all_cameras/all_images for evaluation)
    if all_cameras[0]["width"] > 1600:
        ds = 2
        if all_cameras[0]["width"] > 3200:
            ds = 4
        for cam in all_cameras:
            cam["width"] //= ds
            cam["height"] //= ds
            cam["K"][:2] /= ds
        logger.info(f"Auto-downscaling by {ds}x on top of {downscale}x")

    # Restrict to train views
    trainer.set_training_views(train_indices)

    # Init params + strategy
    trainer.init_params(points3d, colors3d)
    trainer.init_strategy()

    # Train
    t_start = time.time()
    callbacks = TrainerCallbacks(
        progress=lambda frac: logger.info(
            f"  Progress: {frac * 100:.0f}%"
        ) if frac in (0.25, 0.5, 0.75, 1.0) else None,
    )

    output_dir = results_dir / f"gsplat-{config.mode}" / dataset / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "model.ply"

    n_gaussians = trainer.train(callbacks, output_path=ply_path)
    training_time = time.time() - t_start

    # Export PLY
    trainer.export(ply_path)

    logger.info(f"Training complete in {training_time:.1f}s, {n_gaussians} Gaussians")

    # Evaluate on test views
    logger.info("Evaluating on test views...")
    renders_dir = output_dir / "renders"
    eval_results = evaluate_model(
        trainer, test_indices, all_cameras, all_images, save_dir=renders_dir
    )
    eval_results.print_table()

    # Build result dict
    result = {
        "method": f"gsplat-{config.mode}",
        "dataset": dataset,
        "scene": scene_name,
        "config": asdict(config),
        "downscale": downscale,
        "training_time_sec": round(training_time, 1),
        "num_gaussians": n_gaussians,
        "metrics": {
            "mean_psnr": eval_results.mean_psnr,
            "mean_ssim": eval_results.mean_ssim,
            "mean_lpips": eval_results.mean_lpips,
        },
        "per_view": eval_results.per_view,
    }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Cleanup
    trainer.cleanup()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run gsplat benchmark on Mip-NeRF 360 scenes"
    )

    # Scene selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", type=str, help="Single scene name")
    group.add_argument(
        "--all-indoor", action="store_true", help="Run all 4 indoor scenes"
    )
    group.add_argument(
        "--all-outdoor", action="store_true", help="Run all 5 outdoor scenes"
    )
    group.add_argument(
        "--all", action="store_true", help="Run all 9 scenes"
    )

    # Training config
    parser.add_argument("--mode", default="2dgs", choices=["3dgs", "2dgs"])
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--downscale", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--test-every", type=int, default=8)

    # Paths
    parser.add_argument("--dataset", default="mipnerf360")
    parser.add_argument(
        "--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )

    args = parser.parse_args()

    # Determine scenes to run
    if args.scene:
        scenes = [args.scene]
    elif args.all_indoor:
        scenes = MIPNERF360_INDOOR
    elif args.all_outdoor:
        scenes = MIPNERF360_OUTDOOR
    else:
        scenes = MIPNERF360_ALL

    # Build config
    config = TrainerConfig(
        iterations=args.iterations,
        sh_degree=args.sh_degree,
        mode=args.mode,
    )

    all_results = []
    for scene_name in scenes:
        result = run_scene(
            scene_name=scene_name,
            dataset=args.dataset,
            benchmarks_dir=args.benchmarks_dir,
            results_dir=args.results_dir,
            config=config,
            downscale=args.downscale,
            test_every=args.test_every,
        )
        all_results.append(result)

    # Print summary
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"{'Scene':<15} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Time':>8}")
        print("-" * 55)
        for r in all_results:
            m = r["metrics"]
            t = r["training_time_sec"]
            mins = int(t // 60)
            secs = int(t % 60)
            print(
                f"{r['scene']:<15} {m['mean_psnr']:>8.2f} "
                f"{m['mean_ssim']:>8.4f} {m['mean_lpips']:>8.4f} "
                f"{mins:>4}m{secs:02d}s"
            )

        # Compute means
        mean_psnr = sum(r["metrics"]["mean_psnr"] for r in all_results) / len(all_results)
        mean_ssim = sum(r["metrics"]["mean_ssim"] for r in all_results) / len(all_results)
        mean_lpips = sum(r["metrics"]["mean_lpips"] for r in all_results) / len(all_results)
        print("-" * 55)
        print(f"{'Mean':<15} {mean_psnr:>8.2f} {mean_ssim:>8.4f} {mean_lpips:>8.4f}")


if __name__ == "__main__":
    main()
