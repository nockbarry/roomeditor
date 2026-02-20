"""Integration tests — short training runs with real COLMAP data."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

# Path to a project with completed COLMAP reconstruction
TEST_PROJECT = Path("/home/nock/projects/roomeditor/server/data/projects/2aacac41-7971-46e9-ac32-54e6d4eb6fe3")
COLMAP_DIR = TEST_PROJECT / "colmap" / "sparse" / "0"
FRAMES_DIR = TEST_PROJECT / "frames"

pytestmark = pytest.mark.skipif(
    not COLMAP_DIR.exists() or not FRAMES_DIR.exists(),
    reason="Test project data not available",
)


class TestShortTraining3DGS:
    """Short 3DGS training run (50 steps) to verify the full pipeline."""

    def test_train_50_steps(self):
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        config = TrainerConfig(
            iterations=50,
            sh_degree=2,
            mode="3dgs",
            depth_reg_weight=0.1,
        )
        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)

        assert len(trainer.cameras_data) > 0
        assert len(points3d) > 0

        trainer.init_params(points3d, colors3d)
        trainer.init_strategy()

        progress_values = []
        callbacks = TrainerCallbacks(
            progress=lambda x: progress_values.append(x),
            preview_every=25,
            snapshot_every=100,  # Won't fire in 50 steps
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.ply"
            n_final = trainer.train(callbacks, output_path=out)

            assert n_final > 0
            assert len(progress_values) > 0

            # Export
            trainer.export(out)
            assert out.exists()
            assert out.stat().st_size > 1000  # Not trivially empty

        trainer.cleanup()


class TestShortTraining2DGS:
    """Short 2DGS training run (50 steps) to verify 2DGS mode."""

    def test_train_50_steps_2dgs(self):
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        config = TrainerConfig(
            iterations=50,
            sh_degree=2,
            mode="2dgs",
            depth_reg_weight=0.1,
            distortion_weight=0.05,
            normal_weight=0.01,
        )
        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
        trainer.init_params(points3d, colors3d)
        trainer.init_strategy()

        callbacks = TrainerCallbacks(preview_every=100, snapshot_every=200)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_2dgs.ply"
            n_final = trainer.train(callbacks, output_path=out)
            assert n_final > 0

            trainer.export(out)
            assert out.exists()

        trainer.cleanup()


class TestAppearanceEmbeddings:
    """Test training with appearance embeddings enabled."""

    def test_train_with_appearance(self):
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        config = TrainerConfig(
            iterations=30,
            mode="3dgs",
            appearance_embeddings=True,
        )
        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
        trainer.init_params(points3d, colors3d)

        assert trainer.appearance_embeds is not None
        assert trainer.appearance_mlp is not None

        trainer.init_strategy()

        callbacks = TrainerCallbacks(preview_every=100, snapshot_every=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_appear.ply"
            n_final = trainer.train(callbacks, output_path=out)
            assert n_final > 0

        trainer.cleanup()


class TestPreviewCallback:
    """Test that preview callbacks produce valid JPEG images."""

    def test_preview_produces_jpeg(self):
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        config = TrainerConfig(iterations=10, mode="3dgs")
        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
        trainer.init_params(points3d, colors3d)
        trainer.init_strategy()

        previews = []

        def on_preview(step, total, loss, n_gs, jpeg_bytes):
            previews.append({
                "step": step,
                "n_gs": n_gs,
                "size": len(jpeg_bytes),
                "is_jpeg": jpeg_bytes[:2] == b'\xff\xd8',
            })

        callbacks = TrainerCallbacks(
            preview=on_preview,
            preview_every=1,  # Every step for testing
            snapshot_every=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_preview.ply"
            trainer.train(callbacks, output_path=out)

        assert len(previews) > 0
        for p in previews:
            assert p["is_jpeg"]
            assert p["size"] > 100

        trainer.cleanup()


class TestCheckpointResume:
    """Test checkpoint save + load + resume."""

    def test_checkpoint_round_trip(self):
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        config = TrainerConfig(iterations=30, mode="3dgs")
        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
        trainer.init_params(points3d, colors3d)
        trainer.init_strategy()

        callbacks = TrainerCallbacks(preview_every=100, snapshot_every=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_ckpt.ply"
            trainer.train(callbacks, output_path=out)
            assert trainer.current_step == 30

            # Save checkpoint
            ckpt = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(ckpt)
            assert ckpt.exists()

            means_before = trainer.params["means"].data.clone()

            # Load into fresh trainer and verify
            trainer2 = GaussianTrainer(config)
            points3d2, colors3d2 = trainer2.load_data(COLMAP_DIR, FRAMES_DIR)
            trainer2.init_params(points3d2, colors3d2)
            trainer2.init_strategy()
            trainer2.load_checkpoint(ckpt)

            assert trainer2.current_step == 30
            torch.testing.assert_close(
                trainer2.params["means"].data, means_before
            )

        trainer.cleanup()
        trainer2.cleanup()


class TestTIDIPruning:
    """Test TIDI-GS floater pruning in a training run.

    Note: Actual pruning may not fire in 50 steps (needs >1000),
    but we verify the pruner is wired in and doesn't crash.
    """

    def test_train_with_pruning_no_crash(self):
        from pipeline.train_gaussians import (
            GaussianTrainer,
            TrainerConfig,
            TrainerCallbacks,
        )

        config = TrainerConfig(
            iterations=50,
            mode="3dgs",
            tidi_pruning=True,
        )
        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
        trainer.init_params(points3d, colors3d)
        trainer.init_strategy()

        assert trainer.pruner is not None

        callbacks = TrainerCallbacks(preview_every=100, snapshot_every=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_tidi.ply"
            n_final = trainer.train(callbacks, output_path=out)
            assert n_final > 0
            # Pruner should have been updated
            assert trainer.pruner.total_views == 50

        trainer.cleanup()
