"""Full pipeline test on kitchen video using new GaussianTrainer.

Runs: 2DGS Balanced preset with appearance embeddings + TIDI pruning.
Reuses existing COLMAP reconstruction, only re-trains Gaussians.
"""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("kitchen_test")

PROJECT_DIR = Path("/home/nock/projects/roomeditor/server/data/projects/d7a93e8d-a6fa-46eb-b1da-199aba06559a")
COLMAP_DIR = PROJECT_DIR / "colmap" / "sparse" / "0"
FRAMES_DIR = PROJECT_DIR / "frames"
OUTPUT_PLY = PROJECT_DIR / "scene_v2_2dgs.ply"


def main():
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig, TrainerCallbacks

    # 2DGS Balanced preset with appearance + TIDI
    config = TrainerConfig(
        iterations=15_000,
        sh_degree=2,
        mode="2dgs",
        depth_reg_weight=0.2,
        opacity_reg_weight=0.01,
        scale_reg_weight=0.01,
        flatten_reg_weight=0.0,
        distortion_weight=0.1,
        normal_weight=0.01,
        prune_opa=0.01,
        densify_until_pct=0.55,
        appearance_embeddings=True,
        tidi_pruning=True,
    )

    logger.info("=" * 60)
    logger.info("Kitchen Pipeline Test — 2DGS Balanced + Appearance + TIDI")
    logger.info("=" * 60)
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Iterations: {config.iterations}")
    logger.info(f"Appearance embeddings: {config.appearance_embeddings}")
    logger.info(f"TIDI pruning: {config.tidi_pruning}")
    logger.info(f"Distortion weight: {config.distortion_weight}")
    logger.info(f"Normal weight: {config.normal_weight}")

    trainer = GaussianTrainer(config)

    # Load data
    t0 = time.time()
    points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
    t_load = time.time() - t0
    logger.info(f"Data loaded in {t_load:.1f}s: {len(trainer.cameras_data)} views, {len(points3d)} points")

    # Init
    trainer.init_params(points3d, colors3d)
    trainer.init_strategy()

    logger.info(f"Appearance model: embed={trainer.appearance_embeds is not None}, mlp={trainer.appearance_mlp is not None}")
    logger.info(f"Pruner: {trainer.pruner is not None}")

    # Track metrics
    preview_count = 0
    snapshot_count = 0

    def on_progress(frac):
        pass  # Logging handled inside trainer

    def on_preview(step, total, loss, n_gs, jpeg_bytes):
        nonlocal preview_count
        preview_count += 1
        if preview_count % 5 == 0:
            logger.info(f"  Preview #{preview_count}: step={step}, loss={loss:.4f}, N={n_gs}, jpeg={len(jpeg_bytes)}B")

    def on_snapshot(step, total, loss, n_gs, filename):
        nonlocal snapshot_count
        snapshot_count += 1
        logger.info(f"  Snapshot #{snapshot_count}: {filename}, N={n_gs}")

    callbacks = TrainerCallbacks(
        progress=on_progress,
        preview=on_preview,
        snapshot=on_snapshot,
        preview_every=500,
        snapshot_every=2000,
    )

    # Train
    t0 = time.time()
    n_final = trainer.train(callbacks, output_path=OUTPUT_PLY)
    t_train = time.time() - t0

    # Export
    trainer.export(OUTPUT_PLY)

    # Report
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Training time: {t_train:.1f}s ({t_train/60:.1f} min)")
    logger.info(f"Final Gaussians: {n_final:,}")
    logger.info(f"Previews generated: {preview_count}")
    logger.info(f"Snapshots generated: {snapshot_count}")
    logger.info(f"Output: {OUTPUT_PLY} ({OUTPUT_PLY.stat().st_size / 1024 / 1024:.1f} MB)")

    # Check for checkpoints
    ckpts = list(PROJECT_DIR.glob("checkpoint_*.pt"))
    logger.info(f"Checkpoints saved: {len(ckpts)}")
    for c in sorted(ckpts):
        logger.info(f"  {c.name} ({c.stat().st_size / 1024 / 1024:.1f} MB)")

    # Pruner stats
    if trainer.pruner:
        logger.info(f"Pruner total_views: {trainer.pruner.total_views}")

    trainer.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()
