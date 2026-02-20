"""Tests for pipeline.evaluate module — train/test split, metrics, EvalResults."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest


class TestSplitTrainTest:
    def test_default_every_8(self):
        from pipeline.evaluate import split_train_test

        train, test = split_train_test(24)
        assert len(train) + len(test) == 24
        assert set(train) | set(test) == set(range(24))
        assert set(train) & set(test) == set()
        # Every 8th should be test: 0, 8, 16
        assert test == [0, 8, 16]

    def test_custom_interval(self):
        from pipeline.evaluate import split_train_test

        train, test = split_train_test(10, test_every=5)
        assert test == [0, 5]
        assert len(train) == 8

    def test_single_image(self):
        from pipeline.evaluate import split_train_test

        train, test = split_train_test(1)
        assert test == [0]
        assert train == []

    def test_sorted_output(self):
        from pipeline.evaluate import split_train_test

        train, test = split_train_test(100)
        assert train == sorted(train)
        assert test == sorted(test)


class TestComputePSNR:
    def test_identical_images(self):
        from pipeline.evaluate import compute_psnr

        img = torch.rand(64, 64, 3)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images(self):
        from pipeline.evaluate import compute_psnr

        a = torch.zeros(64, 64, 3)
        b = torch.ones(64, 64, 3)
        psnr = compute_psnr(a, b)
        assert psnr == pytest.approx(0.0, abs=0.01)

    def test_reasonable_range(self):
        from pipeline.evaluate import compute_psnr

        a = torch.rand(64, 64, 3)
        b = a + torch.randn_like(a) * 0.05
        b = b.clamp(0, 1)
        psnr = compute_psnr(a, b)
        assert 20.0 < psnr < 50.0


class TestComputeSSIM:
    def test_identical_images(self):
        from pipeline.evaluate import compute_ssim

        img = torch.rand(64, 64, 3)
        ssim = compute_ssim(img, img)
        assert ssim == pytest.approx(1.0, abs=0.001)

    def test_different_images(self):
        from pipeline.evaluate import compute_ssim

        a = torch.zeros(64, 64, 3)
        b = torch.ones(64, 64, 3)
        ssim = compute_ssim(a, b)
        assert ssim < 0.1

    def test_range(self):
        from pipeline.evaluate import compute_ssim

        a = torch.rand(64, 64, 3)
        b = a + torch.randn_like(a) * 0.1
        b = b.clamp(0, 1)
        ssim = compute_ssim(a, b)
        assert 0.0 <= ssim <= 1.0


class TestEvalResults:
    def test_to_json(self):
        from pipeline.evaluate import EvalResults

        results = EvalResults(
            per_view=[{"name": "test", "psnr": 25.0, "ssim": 0.9, "lpips": 0.1}],
            mean_psnr=25.0,
            mean_ssim=0.9,
            mean_lpips=0.1,
            num_test_views=1,
            num_gaussians=1000,
        )
        d = results.to_json()
        assert isinstance(d, dict)
        assert d["mean_psnr"] == 25.0
        assert len(d["per_view"]) == 1

    def test_save_load_json(self):
        from pipeline.evaluate import EvalResults

        results = EvalResults(
            per_view=[
                {"name": "view_000", "index": 0, "psnr": 28.5, "ssim": 0.85, "lpips": 0.15},
                {"name": "view_008", "index": 8, "psnr": 26.3, "ssim": 0.82, "lpips": 0.18},
            ],
            mean_psnr=27.4,
            mean_ssim=0.835,
            mean_lpips=0.165,
            num_test_views=2,
            num_gaussians=50000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results.save_json(path)

            loaded = EvalResults.load_json(path)
            assert loaded.mean_psnr == 27.4
            assert loaded.num_test_views == 2
            assert len(loaded.per_view) == 2
            assert loaded.per_view[0]["name"] == "view_000"

    def test_json_roundtrip(self):
        from pipeline.evaluate import EvalResults

        results = EvalResults(
            per_view=[],
            mean_psnr=30.0,
            mean_ssim=0.95,
            mean_lpips=0.05,
            num_test_views=0,
            num_gaussians=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            results.save_json(path)

            with open(path) as f:
                raw = json.load(f)
            assert raw["mean_psnr"] == 30.0

            loaded = EvalResults.load_json(path)
            assert loaded.to_json() == results.to_json()
