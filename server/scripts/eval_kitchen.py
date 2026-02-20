#!/usr/bin/env python3
"""Quick evaluation test on the kitchen project.

Trains a fresh model with train/test split and evaluates on held-out views.
Uses the existing kitchen project data (d7a93e8d).

Usage:
    python scripts/eval_kitchen.py [--iterations 7000] [--mode 2dgs]
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

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

PROJECT_ID = "d7a93e8d-a6fa-46eb-b1da-199aba06559a"
PROJECT_DIR = Path(__file__).parent.parent / "data" / "projects" / PROJECT_ID


def main():
    parser = argparse.ArgumentParser(description="Evaluate on kitchen project")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--mode", default="2dgs", choices=["3dgs", "2dgs"])
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--test-every", type=int, default=8)
    args = parser.parse_args()

    colmap_dir = PROJECT_DIR / "colmap" / "sparse" / "0"
    images_dir = PROJECT_DIR / "frames"
    output_dir = PROJECT_DIR / "eval" / f"{args.mode}_{args.iterations}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Kitchen evaluation: mode={args.mode}, iterations={args.iterations}")
    logger.info(f"Project: {PROJECT_DIR}")

    # Load all data
    all_cameras, points3d, colors3d = load_colmap_data(colmap_dir, images_dir)

    logger.info("Loading images to GPU...")
    all_images = []
    for cam in all_cameras:
        all_images.append(
            _load_image(cam["image_path"], cam["width"], cam["height"])
        )

    # Split train/test
    n_images = len(all_cameras)
    train_indices, test_indices = split_train_test(n_images, args.test_every)
    logger.info(
        f"Total: {n_images} images -> {len(train_indices)} train, "
        f"{len(test_indices)} test"
    )

    # Create and configure trainer
    config = TrainerConfig(
        iterations=args.iterations,
        sh_degree=args.sh_degree,
        mode=args.mode,
    )
    trainer = GaussianTrainer(config)
    trainer.cameras_data = list(all_cameras)  # copy full list
    trainer.gt_images = list(all_images)      # copy full list

    # Restrict to train views
    trainer.set_training_views(train_indices)

    # Init
    trainer.init_params(points3d, colors3d)
    trainer.init_strategy()

    # Train
    logger.info(f"Training {args.mode} for {args.iterations} iterations...")
    t_start = time.time()

    callbacks = TrainerCallbacks()
    ply_path = output_dir / "model.ply"
    n_gaussians = trainer.train(callbacks, output_path=ply_path)
    training_time = time.time() - t_start

    trainer.export(ply_path)
    logger.info(
        f"Training done: {n_gaussians} Gaussians in {training_time:.1f}s"
    )

    # Evaluate on held-out test views
    logger.info("Evaluating on test views...")
    renders_dir = output_dir / "renders"
    eval_results = evaluate_model(
        trainer, test_indices, all_cameras, all_images, save_dir=renders_dir
    )
    eval_results.print_table()

    # Save results
    result = {
        "method": f"gsplat-{args.mode}",
        "dataset": "kitchen-video",
        "scene": "kitchen",
        "config": asdict(config),
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
    logger.info(f"Rendered test views saved to {renders_dir}")

    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
