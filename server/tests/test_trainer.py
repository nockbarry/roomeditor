"""Tests for pipeline.train_gaussians module — TrainerConfig, TrainerCallbacks, GaussianTrainer."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import pytest


@pytest.fixture
def device():
    return torch.device("cuda")


class TestTrainerConfig:
    def test_defaults(self):
        from pipeline.train_gaussians import TrainerConfig
        cfg = TrainerConfig()
        assert cfg.iterations == 15_000
        assert cfg.sh_degree == 2
        assert cfg.mode == "3dgs"
        assert cfg.appearance_embeddings is False
        assert cfg.tidi_pruning is False

    def test_custom(self):
        from pipeline.train_gaussians import TrainerConfig
        cfg = TrainerConfig(
            iterations=5000,
            mode="2dgs",
            distortion_weight=0.1,
            appearance_embeddings=True,
        )
        assert cfg.iterations == 5000
        assert cfg.mode == "2dgs"
        assert cfg.distortion_weight == 0.1

    def test_serializable(self):
        from pipeline.train_gaussians import TrainerConfig
        cfg = TrainerConfig(iterations=1000)
        d = asdict(cfg)
        assert d["iterations"] == 1000
        assert isinstance(d, dict)


class TestTrainerCallbacks:
    def test_defaults(self):
        from pipeline.train_gaussians import TrainerCallbacks
        cb = TrainerCallbacks()
        assert cb.progress is None
        assert cb.preview is None
        assert cb.snapshot is None
        assert cb.preview_every == 500
        assert cb.snapshot_every == 2000

    def test_custom(self):
        from pipeline.train_gaussians import TrainerCallbacks
        called = []
        cb = TrainerCallbacks(
            progress=lambda x: called.append(x),
            preview_every=100,
        )
        cb.progress(0.5)
        assert called == [0.5]


class TestGaussianTrainer:
    def test_init_3dgs(self):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        from pipeline.rasterizer import Rasterizer3DGS
        trainer = GaussianTrainer(TrainerConfig(mode="3dgs"))
        assert isinstance(trainer.rasterizer, Rasterizer3DGS)

    def test_init_2dgs(self):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        from pipeline.rasterizer import Rasterizer2DGS
        trainer = GaussianTrainer(TrainerConfig(mode="2dgs"))
        assert isinstance(trainer.rasterizer, Rasterizer2DGS)

    def test_init_params_from_numpy(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(iterations=10))
        # Fake camera data for appearance model
        trainer.cameras_data = [{"image_path": "x", "K": np.eye(3), "viewmat": np.eye(4), "width": 64, "height": 64}]

        points = np.random.randn(50, 3).astype(np.float32)
        colors = np.random.rand(50, 3).astype(np.float32)
        trainer.init_params(points, colors)

        assert "means" in trainer.params
        assert "scales" in trainer.params
        assert "quats" in trainer.params
        assert "opacities" in trainer.params
        assert "sh0" in trainer.params
        assert "shN" in trainer.params
        assert trainer.params["means"].shape == (50, 3)
        assert trainer.params["quats"].shape == (50, 4)
        assert trainer.params["sh0"].shape == (50, 1, 3)

    def test_init_params_with_appearance(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(
            iterations=10, appearance_embeddings=True
        ))
        trainer.cameras_data = [
            {"image_path": f"img{i}", "K": np.eye(3), "viewmat": np.eye(4), "width": 64, "height": 64}
            for i in range(5)
        ]
        points = np.random.randn(20, 3).astype(np.float32)
        colors = np.random.rand(20, 3).astype(np.float32)
        trainer.init_params(points, colors)

        assert trainer.appearance_embeds is not None
        assert trainer.appearance_mlp is not None
        assert "appearance_embeds" in trainer.optimizers
        assert "appearance_mlp" in trainer.optimizers

    def test_init_strategy_3dgs(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(mode="3dgs", iterations=100))
        trainer.cameras_data = [{"image_path": "x"}]
        points = np.random.randn(20, 3).astype(np.float32)
        colors = np.random.rand(20, 3).astype(np.float32)
        trainer.init_params(points, colors)
        trainer.init_strategy()
        assert trainer.strategy is not None
        assert trainer.strategy_state is not None
        assert trainer.pruner is None

    def test_init_strategy_2dgs(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(mode="2dgs", iterations=100))
        trainer.cameras_data = [{"image_path": "x"}]
        points = np.random.randn(20, 3).astype(np.float32)
        colors = np.random.rand(20, 3).astype(np.float32)
        trainer.init_params(points, colors)
        trainer.init_strategy()
        assert trainer.strategy is not None

    def test_init_strategy_with_pruner(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(
            iterations=100, tidi_pruning=True
        ))
        trainer.cameras_data = [{"image_path": "x"}]
        points = np.random.randn(20, 3).astype(np.float32)
        colors = np.random.rand(20, 3).astype(np.float32)
        trainer.init_params(points, colors)
        trainer.init_strategy()
        assert trainer.pruner is not None

    def test_checkpoint_save_load(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(iterations=100))
        trainer.cameras_data = [{"image_path": "x"}]
        points = np.random.randn(30, 3).astype(np.float32)
        colors = np.random.rand(30, 3).astype(np.float32)
        trainer.init_params(points, colors)
        trainer.init_strategy()
        trainer.current_step = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            # Load into new trainer
            trainer2 = GaussianTrainer(TrainerConfig(iterations=100))
            trainer2.cameras_data = [{"image_path": "x"}]
            points2 = np.random.randn(30, 3).astype(np.float32)
            colors2 = np.random.rand(30, 3).astype(np.float32)
            trainer2.init_params(points2, colors2)
            trainer2.init_strategy()
            trainer2.load_checkpoint(ckpt_path)

            assert trainer2.current_step == 42
            # Params should match saved values
            torch.testing.assert_close(
                trainer2.params["means"].data,
                trainer.params["means"].data,
            )

    def test_checkpoint_save_load_with_appearance(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(
            iterations=100, appearance_embeddings=True
        ))
        trainer.cameras_data = [
            {"image_path": f"img{i}", "K": np.eye(3), "viewmat": np.eye(4), "width": 64, "height": 64}
            for i in range(3)
        ]
        points = np.random.randn(20, 3).astype(np.float32)
        colors = np.random.rand(20, 3).astype(np.float32)
        trainer.init_params(points, colors)
        trainer.init_strategy()
        trainer.current_step = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt_appear.pt"
            trainer.save_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            # Verify checkpoint contains appearance state
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            assert "appearance_embeds" in ckpt
            assert "appearance_mlp" in ckpt

    def test_prune_gaussians(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(iterations=100))
        trainer.cameras_data = [{"image_path": "x"}]
        points = np.random.randn(50, 3).astype(np.float32)
        colors = np.random.rand(50, 3).astype(np.float32)
        trainer.init_params(points, colors)

        mask = torch.zeros(50, dtype=torch.bool, device=device)
        mask[10:20] = True  # Prune 10 Gaussians
        trainer._prune_gaussians(mask)

        assert len(trainer.params["means"]) == 40
        assert len(trainer.params["scales"]) == 40
        assert len(trainer.params["opacities"]) == 40

    def test_export(self, device):
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
        trainer = GaussianTrainer(TrainerConfig(iterations=10))
        trainer.cameras_data = [{"image_path": "x"}]
        points = np.random.randn(30, 3).astype(np.float32)
        colors = np.random.rand(30, 3).astype(np.float32)
        trainer.init_params(points, colors)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.ply"
            trainer.export(out)
            assert out.exists()
            assert out.stat().st_size > 0
