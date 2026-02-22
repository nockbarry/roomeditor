#!/usr/bin/env python3
"""End-to-end test of the AnySplat → Refinement pipeline.

Tests the full reconstruction + refinement pipeline using Mip-NeRF 360 benchmark
scenes. Measures PSNR/SSIM/LPIPS at each stage to verify quality improvements.

Usage:
    # Quick test (~2 min, 16 views at 224px)
    python scripts/test_refine_pipeline.py --quick

    # Standard test (~10 min, 32 views at 448px)
    python scripts/test_refine_pipeline.py

    # Full test (~30 min, all presets, LPIPS eval)
    python scripts/test_refine_pipeline.py --full

    # Custom scene
    python scripts/test_refine_pipeline.py --scene kitchen --views 24
"""

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_refine")

# Mip-NeRF 360 benchmark data
BENCHMARKS_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "mipnerf360"
# Test output directory
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test_refine_pipeline"


@dataclass
class StageMetrics:
    """Metrics for one pipeline stage."""
    stage: str
    psnr: float
    ssim: float
    lpips: float | None
    n_gaussians: int
    duration_sec: float
    extra: dict | None = None


def load_benchmark_images(
    scene_dir: Path, downscale: int, max_views: int
) -> tuple[list[Path], list[int]]:
    """Load image paths from a benchmark scene, subsampled to max_views."""
    if downscale > 1:
        images_dir = scene_dir / f"images_{downscale}"
    else:
        images_dir = scene_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    paths = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Subsample evenly if needed
    if len(paths) > max_views:
        step = len(paths) / max_views
        indices = [int(i * step) for i in range(max_views)]
        paths = [paths[i] for i in indices]
    else:
        indices = list(range(len(paths)))

    return paths, indices


def prepare_test_project(
    scene_name: str, image_paths: list[Path], output_dir: Path
) -> Path:
    """Create a test project directory with symlinked frames."""
    project_dir = output_dir / scene_name
    frames_dir = project_dir / "frames"

    # Clean previous run
    if project_dir.exists():
        shutil.rmtree(project_dir)

    frames_dir.mkdir(parents=True)

    # Symlink images as frames
    for i, img_path in enumerate(image_paths):
        dst = frames_dir / f"frame_{i:05d}.jpg"
        # Copy instead of symlink (AnySplat needs real files)
        shutil.copy2(str(img_path), str(dst))

    logger.info(f"Prepared test project: {len(image_paths)} frames in {project_dir}")
    return project_dir


def run_anysplat(
    project_dir: Path, max_views: int, resolution: int
) -> tuple[float, int]:
    """Run AnySplat on the test project. Returns (duration_sec, n_gaussians)."""
    from pipeline.run_anysplat_subprocess import run_anysplat_subprocess

    frames_dir = project_dir / "frames"
    output_ply = project_dir / "scene.ply"

    logger.info(f"Running AnySplat: {max_views} views @ {resolution}px...")
    start = time.time()
    n_gaussians = run_anysplat_subprocess(
        images_dir=frames_dir,
        output_ply=output_ply,
        max_views=max_views,
        resolution=resolution,
    )
    duration = time.time() - start
    logger.info(f"AnySplat done: {n_gaussians:,} gaussians in {duration:.1f}s")
    return duration, n_gaussians


def evaluate_ply(
    project_dir: Path,
    trainer_or_params=None,
    test_indices: list[int] | None = None,
    all_cameras: list[dict] | None = None,
    all_images: list[torch.Tensor] | None = None,
    compute_lpips: bool = False,
) -> dict:
    """Evaluate a PLY file's quality by rendering held-out test views.

    Returns dict with mean_psnr, mean_ssim, and optionally mean_lpips.
    """
    from pipeline.evaluate import (
        compute_psnr,
        compute_ssim,
        compute_lpips as _compute_lpips,
        _init_lpips,
    )

    device = torch.device("cuda")
    psnrs, ssims, lpipss = [], [], []

    lpips_net = None
    if compute_lpips:
        try:
            lpips_net = _init_lpips(device)
        except Exception:
            logger.warning("LPIPS unavailable, skipping")

    with torch.no_grad():
        for idx in test_indices:
            cam = all_cameras[idx]
            gt = all_images[idx]
            W, H = cam["width"], cam["height"]

            viewmat = torch.tensor(cam["viewmat"], device=device).unsqueeze(0)
            K = torch.tensor(cam["K"], device=device).unsqueeze(0)

            result = trainer_or_params.rasterizer.rasterize(
                trainer_or_params.params, viewmat, K, W, H,
                trainer_or_params.config.sh_degree,
            )
            rendered = result.image.clamp(0, 1)

            psnrs.append(compute_psnr(rendered, gt))
            ssims.append(compute_ssim(rendered, gt))
            if lpips_net is not None:
                lpipss.append(_compute_lpips(rendered, gt, lpips_net))

    if lpips_net is not None:
        del lpips_net
        torch.cuda.empty_cache()

    result = {
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
    }
    if lpipss:
        result["mean_lpips"] = float(np.mean(lpipss))
    return result


def run_refine_and_eval(
    project_dir: Path,
    preset: str,
    all_cameras: list[dict],
    all_images: list[torch.Tensor],
    train_indices: list[int],
    test_indices: list[int],
    compute_lpips: bool = False,
) -> StageMetrics:
    """Run a single refinement preset and evaluate quality."""
    from plyfile import PlyData
    from app.routes.postprocess import REFINE_PRESETS
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig, TrainerCallbacks

    ply_path = project_dir / "scene.ply"

    # Resolve preset
    preset_def = REFINE_PRESETS[preset]
    preset_config = preset_def["config"]
    iterations = preset_config["iterations"]

    # Load PLY parameters
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n_pts = len(vertex.data)
    field_names = vertex.data.dtype.names

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float32)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1).astype(np.float32)
    opacities = np.array(vertex["opacity"]).astype(np.float32)
    sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)

    config = TrainerConfig(
        iterations=iterations,
        sh_degree=preset_config.get("sh_degree", 1),
        mode="3dgs",
        depth_reg_weight=preset_config.get("depth_reg_weight", 0.05),
        opacity_reg_weight=preset_config.get("opacity_reg_weight", 0.01),
        scale_reg_weight=preset_config.get("scale_reg_weight", 0.01),
        prune_opa=0.005,
        densify_until_pct=0.5,
        means_lr=preset_config.get("means_lr", 1.6e-4),
        means_lr_final=preset_config.get("means_lr_final", 1.6e-6),
        scales_lr=preset_config.get("scales_lr", 5e-3),
        quats_lr=preset_config.get("quats_lr", 1e-3),
        opacities_lr=preset_config.get("opacities_lr", 5e-2),
        sh0_lr=preset_config.get("sh0_lr", 2.5e-3),
        shN_lr=preset_config.get("shN_lr", 2.5e-3),
        densify_enabled=preset_config.get("densify_enabled", True),
        reset_opacity=preset_config.get("reset_opacity", True),
        freeze_positions_pct=preset_config.get("freeze_positions_pct", 0.0),
        grow_grad2d=preset_config.get("grow_grad2d", 0.0002),
    )

    trainer = GaussianTrainer(config)
    trainer.cameras_data = list(all_cameras)
    trainer.gt_images = list(all_images)
    trainer._all_cameras = all_cameras
    trainer._all_images = all_images
    trainer._eval_views = test_indices

    trainer.init_params_from_existing(means, scales, quats, opacities, sh0)
    trainer.init_strategy()
    trainer.set_training_views(train_indices)

    # Track metrics during training
    train_metrics = []
    eval_metrics = []

    def on_metrics(m):
        train_metrics.append(m)

    def on_eval(e):
        eval_metrics.append(e)

    callbacks = TrainerCallbacks(
        metrics=on_metrics,
        evaluation=on_eval,
        metrics_every=50,
        eval_every=500,
    )

    logger.info(f"  Refining with '{preset}' preset ({iterations} iters)...")
    start = time.time()
    output_path = project_dir / f"refined_{preset}.ply"
    n_final = trainer.train(callbacks, output_path)
    trainer.export(output_path)
    duration = time.time() - start
    logger.info(f"  Done in {duration:.1f}s ({n_final:,} gaussians)")

    # Evaluate on test views
    eval_result = evaluate_ply(
        project_dir,
        trainer_or_params=trainer,
        test_indices=test_indices,
        all_cameras=all_cameras,
        all_images=all_images,
        compute_lpips=compute_lpips,
    )

    # Collect training curve info
    extra = {
        "iterations_actual": trainer.current_step,
        "loss_start": train_metrics[0]["losses"]["total"] if train_metrics else None,
        "loss_end": train_metrics[-1]["losses"]["total"] if train_metrics else None,
    }
    if eval_metrics:
        extra["eval_psnr_curve"] = [
            {"step": e["step"], "psnr": e["mean_psnr"]} for e in eval_metrics
        ]

    trainer.cleanup()

    return StageMetrics(
        stage=f"refine_{preset}",
        psnr=eval_result["mean_psnr"],
        ssim=eval_result["mean_ssim"],
        lpips=eval_result.get("mean_lpips"),
        n_gaussians=n_final,
        duration_sec=duration,
        extra=extra,
    )


def evaluate_baseline(
    project_dir: Path,
    all_cameras: list[dict],
    all_images: list[torch.Tensor],
    test_indices: list[int],
    compute_lpips: bool = False,
) -> StageMetrics:
    """Evaluate the AnySplat baseline (before any refinement)."""
    from plyfile import PlyData
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig

    ply_path = project_dir / "scene.ply"
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n_pts = len(vertex.data)

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float32)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1).astype(np.float32)
    opacities = np.array(vertex["opacity"]).astype(np.float32)
    sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)

    # Create a trainer just for evaluation (no training)
    config = TrainerConfig(iterations=0, sh_degree=0, densify_enabled=False)
    trainer = GaussianTrainer(config)
    trainer.cameras_data = all_cameras
    trainer.gt_images = all_images
    trainer.init_params_from_existing(means, scales, quats, opacities, sh0)

    eval_result = evaluate_ply(
        project_dir,
        trainer_or_params=trainer,
        test_indices=test_indices,
        all_cameras=all_cameras,
        all_images=all_images,
        compute_lpips=compute_lpips,
    )

    trainer.cleanup()

    return StageMetrics(
        stage="baseline",
        psnr=eval_result["mean_psnr"],
        ssim=eval_result["mean_ssim"],
        lpips=eval_result.get("mean_lpips"),
        n_gaussians=n_pts,
        duration_sec=0.0,
    )


def load_cameras_and_images(
    project_dir: Path,
) -> tuple[list[dict], list[torch.Tensor]]:
    """Load cameras.json and all frame images into GPU tensors."""
    from PIL import Image

    cameras_path = project_dir / "cameras.json"
    frames_dir = project_dir / "frames"

    with open(cameras_path) as f:
        cam_data = json.load(f)

    resolution = cam_data.get("resolution", 448)
    default_focal = cam_data.get("focal_length_px", resolution * 0.85)

    cameras_data = []
    images = []

    for cam in cam_data["cameras"]:
        frame_path = frames_dir / cam["frame"]
        if not frame_path.exists():
            continue

        transform = np.array(cam["transform"], dtype=np.float32)
        if transform.shape == (3, 4):
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :] = transform
        else:
            c2w = transform

        viewmat = np.linalg.inv(c2w).astype(np.float32)

        fx = cam.get("fx", default_focal)
        fy = cam.get("fy", default_focal)
        cx = cam.get("cx", resolution / 2)
        cy = cam.get("cy", resolution / 2)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        cameras_data.append({
            "image_path": str(frame_path),
            "K": K,
            "viewmat": viewmat,
            "width": resolution,
            "height": resolution,
        })

        img = Image.open(frame_path).convert("RGB")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(torch.tensor(arr, dtype=torch.float32, device="cuda"))

    return cameras_data, images


def print_results_table(stages: list[StageMetrics], baseline: StageMetrics):
    """Print a formatted comparison table."""
    print()
    print("=" * 85)
    print(f"{'REFINEMENT PIPELINE TEST RESULTS':^85}")
    print("=" * 85)

    has_lpips = any(s.lpips is not None for s in stages)

    header = f"{'Stage':<20} {'PSNR (dB)':>10} {'dPSNR':>8} {'SSIM':>8} {'dSSIM':>8}"
    if has_lpips:
        header += f" {'LPIPS':>8}"
    header += f" {'#Gauss':>10} {'Time':>8}"
    print(header)
    print("-" * 85)

    for s in stages:
        delta_psnr = s.psnr - baseline.psnr
        delta_ssim = s.ssim - baseline.ssim
        psnr_color = "+" if delta_psnr >= 0 else ""
        ssim_color = "+" if delta_ssim >= 0 else ""

        row = (
            f"{s.stage:<20} "
            f"{s.psnr:>10.2f} "
            f"{psnr_color}{delta_psnr:>7.2f} "
            f"{s.ssim:>8.4f} "
            f"{ssim_color}{delta_ssim:>7.4f}"
        )
        if has_lpips:
            row += f" {s.lpips:>8.4f}" if s.lpips is not None else f" {'N/A':>8}"
        row += f" {s.n_gaussians:>10,} {s.duration_sec:>7.1f}s"
        print(row)

    print("-" * 85)

    # Summary verdict
    improved = [s for s in stages if s.psnr > baseline.psnr and s.stage != "baseline"]
    degraded = [s for s in stages if s.psnr < baseline.psnr and s.stage != "baseline"]

    if improved:
        best = max(improved, key=lambda s: s.psnr)
        print(f"\nBest improvement: {best.stage} ({best.psnr - baseline.psnr:+.2f} dB)")
    if degraded:
        worst = min(degraded, key=lambda s: s.psnr)
        print(f"Worst degradation: {worst.stage} ({worst.psnr - baseline.psnr:+.2f} dB)")

    # Per-preset training curve
    for s in stages:
        if s.extra and s.extra.get("eval_psnr_curve"):
            curve = s.extra["eval_psnr_curve"]
            curve_str = " → ".join(
                f"{p['step']}:{p['psnr']:.1f}" for p in curve
            )
            print(f"  {s.stage} eval curve: {curve_str}")

    print()


def save_results(stages: list[StageMetrics], output_dir: Path):
    """Save results as JSON for programmatic comparison."""
    results = []
    for s in stages:
        d = {
            "stage": s.stage,
            "psnr": round(s.psnr, 4),
            "ssim": round(s.ssim, 4),
            "n_gaussians": s.n_gaussians,
            "duration_sec": round(s.duration_sec, 1),
        }
        if s.lpips is not None:
            d["lpips"] = round(s.lpips, 4)
        if s.extra:
            d["extra"] = s.extra
        results.append(d)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Test refinement pipeline")
    parser.add_argument("--scene", default="counter", help="Mip-NeRF 360 scene name")
    parser.add_argument("--views", type=int, default=None, help="Max views for AnySplat")
    parser.add_argument("--resolution", type=int, default=None, help="Override AnySplat resolution")
    parser.add_argument("--downscale", type=int, default=8, help="Image downscale factor (1,2,4,8)")
    parser.add_argument("--presets", nargs="+", default=None, help="Presets to test")
    parser.add_argument("--quick", action="store_true", help="Quick test: 16 views, conservative only")
    parser.add_argument("--full", action="store_true", help="Full test: all presets + LPIPS")
    parser.add_argument("--skip-anysplat", action="store_true", help="Skip AnySplat, use existing scene.ply")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS (slower)")
    args = parser.parse_args()

    # Apply presets
    if args.quick:
        views = args.views or 16
        resolution = args.resolution or 224
        presets = ["conservative"]
        compute_lpips = False
        downscale = 8
    elif args.full:
        views = args.views or 32
        resolution = args.resolution or 448
        presets = ["conservative", "balanced", "aggressive"]
        compute_lpips = True
        downscale = args.downscale
    else:
        views = args.views or 24
        resolution = args.resolution or 448
        presets = args.presets or ["conservative", "balanced"]
        compute_lpips = args.lpips
        downscale = args.downscale

    scene_name = args.scene
    scene_dir = BENCHMARKS_DIR / scene_name

    if not scene_dir.exists():
        logger.error(f"Scene not found: {scene_dir}")
        logger.info(f"Available scenes: {[d.name for d in BENCHMARKS_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)

    logger.info(f"Pipeline test: scene={scene_name}, views={views}, res={resolution}, "
                f"presets={presets}, downscale={downscale}, lpips={compute_lpips}")

    # 1. Prepare test project from benchmark images
    output_dir = TEST_OUTPUT_DIR / scene_name
    image_paths, _ = load_benchmark_images(scene_dir, downscale, views)
    project_dir = prepare_test_project(scene_name, image_paths, TEST_OUTPUT_DIR)

    all_stages: list[StageMetrics] = []

    # 2. Run AnySplat
    if not args.skip_anysplat or not (project_dir / "scene.ply").exists():
        anysplat_duration, anysplat_n = run_anysplat(project_dir, views, resolution)
    else:
        logger.info("Skipping AnySplat, using existing scene.ply")
        from plyfile import PlyData
        plydata = PlyData.read(str(project_dir / "scene.ply"))
        anysplat_n = len(plydata["vertex"].data)
        anysplat_duration = 0.0

    # 3. Load cameras and images
    logger.info("Loading cameras and images...")
    all_cameras, all_images = load_cameras_and_images(project_dir)
    n_views = len(all_cameras)
    logger.info(f"Loaded {n_views} camera views")

    # Train/test split (every 4th for AnySplat's fewer views)
    from app.routes.postprocess import _split_train_test
    train_indices, test_indices = _split_train_test(n_views, test_every=4)
    logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test views")

    if not test_indices:
        logger.warning("No test views — using all views for both train and eval")
        test_indices = list(range(n_views))

    # 4. Evaluate baseline (AnySplat output, before refinement)
    logger.info("Evaluating baseline (AnySplat output)...")
    baseline = evaluate_baseline(
        project_dir, all_cameras, all_images, test_indices, compute_lpips
    )
    baseline.duration_sec = anysplat_duration
    baseline.n_gaussians = anysplat_n
    all_stages.append(baseline)
    logger.info(f"Baseline: PSNR={baseline.psnr:.2f} dB, SSIM={baseline.ssim:.4f}")

    # 5. Run each refinement preset
    for preset in presets:
        logger.info(f"\n--- Preset: {preset} ---")

        # Restore original scene.ply for each preset (refine from same baseline)
        ply_path = project_dir / "scene.ply"
        backup = project_dir / "scene_baseline.ply"
        if not backup.exists():
            shutil.copy2(str(ply_path), str(backup))
        else:
            shutil.copy2(str(backup), str(ply_path))

        stage = run_refine_and_eval(
            project_dir, preset,
            all_cameras, all_images,
            train_indices, test_indices,
            compute_lpips,
        )
        all_stages.append(stage)
        delta = stage.psnr - baseline.psnr
        logger.info(f"  {preset}: PSNR={stage.psnr:.2f} dB ({delta:+.2f}), SSIM={stage.ssim:.4f}")

    # 6. Print and save results
    print_results_table(all_stages, baseline)
    save_results(all_stages, output_dir)

    # 7. PASS/FAIL verdict
    passing = True
    for s in all_stages:
        if s.stage == "baseline":
            continue
        if s.stage == "refine_conservative" and s.psnr < baseline.psnr:
            logger.error(f"FAIL: {s.stage} degraded PSNR ({s.psnr:.2f} < {baseline.psnr:.2f})")
            passing = False

    if passing:
        logger.info("PASS: All quality checks passed")
    else:
        logger.error("FAIL: Some quality checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
