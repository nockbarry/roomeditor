#!/usr/bin/env python3
"""Run 30k 2DGS training on the kitchen-video project with improved settings.

Trains on the existing COLMAP data, evaluates, saves results to benchmarks dir,
and copies the PLY to a new project.
"""

import json
import logging
import sqlite3
import sys
import time
import uuid
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from pipeline.evaluate import evaluate_model, split_train_test
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

# Paths
SERVER_DIR = Path(__file__).parent.parent
DATA_DIR = SERVER_DIR / "data"
SOURCE_PROJECT = "d7a93e8d-a6fa-46eb-b1da-199aba06559a"
PROJECT_DIR = DATA_DIR / "projects" / SOURCE_PROJECT
COLMAP_DIR = PROJECT_DIR / "colmap" / "sparse" / "0"
IMAGES_DIR = PROJECT_DIR / "frames"

RESULTS_DIR = DATA_DIR / "benchmarks" / "results"
DATASET = "kitchen-video"
SCENE = "kitchen"
TEST_EVERY = 8

# Improved config for 30k
config = TrainerConfig(
    iterations=30_000,
    sh_degree=3,
    mode="2dgs",
    depth_reg_weight=0.1,
    normal_weight=0.05,
    distortion_weight=0.1,
    scale_reg_weight=0.01,
    opacity_reg_weight=0.0,
    flatten_reg_weight=0.0,
    prune_opa=0.005,
    densify_until_pct=0.5,
    appearance_embeddings=False,
    tidi_pruning=False,
)


def main():
    logger.info("Loading COLMAP data...")
    all_cameras, points3d, colors3d = load_colmap_data(COLMAP_DIR, IMAGES_DIR)

    logger.info("Loading images to GPU...")
    all_images = []
    for cam in all_cameras:
        all_images.append(_load_image(cam["image_path"], cam["width"], cam["height"]))

    # Split train/test
    n_images = len(all_cameras)
    train_indices, test_indices = split_train_test(n_images, TEST_EVERY)
    logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test")

    # Auto-downscale large images
    if all_cameras[0]["width"] > 1600:
        ds = 2
        if all_cameras[0]["width"] > 3200:
            ds = 4
        for cam in all_cameras:
            cam["width"] //= ds
            cam["height"] //= ds
            cam["K"][:2] /= ds
        logger.info(f"Auto-downscaling by {ds}x")

    # Create trainer
    trainer = GaussianTrainer(config)
    trainer.cameras_data = all_cameras
    trainer.gt_images = all_images
    trainer.set_training_views(train_indices)
    trainer.init_params(points3d, colors3d)
    trainer.init_strategy()

    # Output dir
    method = f"gsplat-{config.mode}"
    output_dir = RESULTS_DIR / method / DATASET / SCENE
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "model.ply"

    # Train with progress logging
    def on_progress(frac):
        if frac in (0.1, 0.25, 0.5, 0.75, 1.0):
            logger.info(f"  Progress: {frac * 100:.0f}%")

    callbacks = TrainerCallbacks(progress=on_progress)

    logger.info(f"Starting training: {config.iterations} iterations, mode={config.mode}")
    logger.info(f"  sh_degree={config.sh_degree}, normal_weight={config.normal_weight}")
    logger.info(f"  distortion_weight={config.distortion_weight}, scale_reg={config.scale_reg_weight}")

    t_start = time.time()
    n_gaussians = trainer.train(callbacks, output_path=ply_path)
    training_time = time.time() - t_start

    # Export PLY
    trainer.export(ply_path)
    logger.info(f"Training complete in {training_time:.1f}s, {n_gaussians} Gaussians")

    # Evaluate
    logger.info("Evaluating on test views...")
    renders_dir = output_dir / "renders"
    eval_results = evaluate_model(
        trainer, test_indices, all_cameras, all_images, save_dir=renders_dir
    )
    eval_results.print_table()

    # Save results
    result = {
        "method": method,
        "dataset": DATASET,
        "scene": SCENE,
        "config": {
            "iterations": config.iterations,
            "sh_degree": config.sh_degree,
            "mode": config.mode,
            "depth_reg_weight": config.depth_reg_weight,
            "normal_weight": config.normal_weight,
            "distortion_weight": config.distortion_weight,
            "scale_reg_weight": config.scale_reg_weight,
            "opacity_reg_weight": config.opacity_reg_weight,
            "flatten_reg_weight": config.flatten_reg_weight,
            "prune_opa": config.prune_opa,
            "densify_until_pct": config.densify_until_pct,
            "appearance_embeddings": config.appearance_embeddings,
            "tidi_pruning": config.tidi_pruning,
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

    # Create a new project for the viewer
    new_id = str(uuid.uuid4())
    new_project_dir = DATA_DIR / "projects" / new_id
    new_project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_path, new_project_dir / "scene.ply")

    db_path = SERVER_DIR / "roomeditor.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO projects (id, name, status, gaussian_count, created_at, updated_at) "
        "VALUES (?, ?, 'ready', ?, datetime('now'), datetime('now'))",
        (new_id, "Kitchen Benchmark (2DGS 30k)", n_gaussians),
    )
    conn.commit()
    conn.close()

    logger.info(f"Created project {new_id}: Kitchen Benchmark (2DGS 30k)")
    logger.info(f"View at: http://localhost:5173/project/{new_id}")

    # Cleanup
    trainer.cleanup()
    torch.cuda.empty_cache()

    logger.info("Done!")


if __name__ == "__main__":
    main()
