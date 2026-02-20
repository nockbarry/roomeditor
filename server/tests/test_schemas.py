"""Tests for app.schemas — verify new fields work correctly."""

import pytest


class TestTrainingConfig:
    def test_defaults(self):
        from app.schemas import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.iterations == 15_000
        assert cfg.mode == "3dgs"
        assert cfg.distortion_weight == 0.0
        assert cfg.normal_weight == 0.0
        assert cfg.appearance_embeddings is False
        assert cfg.tidi_pruning is False
        assert cfg.resume_from is None

    def test_2dgs_config(self):
        from app.schemas import TrainingConfig
        cfg = TrainingConfig(
            mode="2dgs",
            distortion_weight=0.1,
            normal_weight=0.01,
            appearance_embeddings=True,
            tidi_pruning=True,
        )
        assert cfg.mode == "2dgs"
        assert cfg.distortion_weight == 0.1
        assert cfg.appearance_embeddings is True

    def test_resume_from(self):
        from app.schemas import TrainingConfig
        cfg = TrainingConfig(resume_from="checkpoint_5000.pt")
        assert cfg.resume_from == "checkpoint_5000.pt"

    def test_model_dump(self):
        from app.schemas import TrainingConfig
        cfg = TrainingConfig(mode="2dgs", appearance_embeddings=True)
        d = cfg.model_dump()
        assert d["mode"] == "2dgs"
        assert d["appearance_embeddings"] is True
        assert "resume_from" in d

    def test_extra_fields_ignored(self):
        from app.schemas import TrainingConfig
        # extra="ignore" should allow unknown fields
        cfg = TrainingConfig(iterations=1000, unknown_field="hello")
        assert cfg.iterations == 1000
