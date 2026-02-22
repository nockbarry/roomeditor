#!/usr/bin/env python3
"""Run the full pipeline on indoor benchmark scenes and create viewable projects.

Loads Mip-NeRF 360 indoor scenes (room, kitchen, counter) through the full
pipeline: frame selection → AnySplat reconstruction → refinement → evaluation.
Creates proper database entries so results are viewable in the UI.

Usage:
    # All 3 indoor scenes, balanced refinement (recommended)
    python scripts/run_indoor_benchmarks.py

    # Single scene, quick test
    python scripts/run_indoor_benchmarks.py --scenes counter --quick

    # All scenes with all presets
    python scripts/run_indoor_benchmarks.py --full

    # Skip AnySplat if already run (re-run refinement only)
    python scripts/run_indoor_benchmarks.py --skip-anysplat

    # Chunked mode: use all available images via chunked pipeline
    python scripts/run_indoor_benchmarks.py --chunked

    # Compare single-pass vs chunked
    python scripts/run_indoor_benchmarks.py --scenes counter --chunked --views 64
"""

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indoor_bench")

# Paths
SERVER_DIR = Path(__file__).parent.parent
DATA_DIR = SERVER_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
BENCHMARKS_DIR = DATA_DIR / "benchmarks" / "mipnerf360"
DB_PATH = SERVER_DIR / "roomeditor.db"

# Indoor scenes most relevant to room editing
INDOOR_SCENES = ["room", "kitchen", "counter"]


@dataclass
class StageResult:
    """Metrics for one pipeline stage."""
    stage: str
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float | None = None
    n_gaussians: int = 0
    duration_sec: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------- Database helpers (sync SQLite, no server needed) ----------

def db_create_project(name: str, project_id: str | None = None) -> str:
    """Insert a project row directly into the SQLite DB."""
    pid = project_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Check if project already exists
        row = conn.execute("SELECT id FROM projects WHERE id = ?", (pid,)).fetchone()
        if row:
            logger.info(f"  Project {pid} already exists, updating")
            conn.execute(
                "UPDATE projects SET name=?, status='processing', updated_at=? WHERE id=?",
                (name, now, pid),
            )
        else:
            conn.execute(
                "INSERT INTO projects (id, name, status, created_at, updated_at) VALUES (?, ?, 'processing', ?, ?)",
                (pid, name, now, now),
            )
        conn.commit()
    finally:
        conn.close()

    # Create directories
    project_dir = PROJECTS_DIR / pid
    (project_dir / "sources").mkdir(parents=True, exist_ok=True)
    (project_dir / "frames").mkdir(parents=True, exist_ok=True)

    return pid


def db_update_project(project_id: str, **kwargs):
    """Update project fields in the DB."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        sets = ["updated_at=?"]
        vals = [now]
        for k, v in kwargs.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(project_id)
        conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def db_create_job(project_id: str, job_type: str = "reconstruct") -> str:
    """Insert a job row."""
    jid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            "INSERT INTO jobs (id, project_id, job_type, status, progress, created_at, updated_at) "
            "VALUES (?, ?, ?, 'completed', 1.0, ?, ?)",
            (jid, project_id, job_type, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return jid


def db_find_project_by_name(name: str) -> str | None:
    """Find existing project by name."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ---------- Image / camera helpers ----------

def select_benchmark_images(
    scene_dir: Path, downscale: int, max_views: int,
) -> list[Path]:
    """Select evenly-spaced images from a benchmark scene."""
    if downscale > 1:
        images_dir = scene_dir / f"images_{downscale}"
    else:
        images_dir = scene_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found: {images_dir}")

    paths = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images in {images_dir}")

    if len(paths) > max_views:
        step = len(paths) / max_views
        paths = [paths[int(i * step)] for i in range(max_views)]

    return paths


def copy_frames_to_project(image_paths: list[Path], project_dir: Path):
    """Copy benchmark images into a project's frames directory."""
    frames_dir = project_dir / "frames"

    # Clear existing frames
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.jpg"):
            f.unlink()

    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        dst = frames_dir / f"frame_{i:05d}.jpg"
        shutil.copy2(str(img_path), str(dst))

    logger.info(f"  Copied {len(image_paths)} frames to {frames_dir}")


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
            "K": K, "viewmat": viewmat,
            "width": resolution, "height": resolution,
        })

        img = Image.open(frame_path).convert("RGB")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(torch.tensor(arr, dtype=torch.float32, device="cuda"))

    return cameras_data, images


# ---------- Pipeline steps ----------

def run_anysplat(
    project_dir: Path, max_views: int, resolution: int,
    chunked: bool = False, chunk_size: int = 32, chunk_overlap: int = 8,
) -> tuple[float, int]:
    """Run AnySplat on a project. Returns (duration, n_gaussians)."""
    from pipeline.run_anysplat_subprocess import run_anysplat_subprocess

    frames_dir = project_dir / "frames"
    output_ply = project_dir / "scene.ply"

    mode_str = f"chunked({chunk_size}/{chunk_overlap})" if chunked else "single-pass"
    logger.info(f"  Running AnySplat: {max_views} views @ {resolution}px [{mode_str}]...")
    start = time.time()
    n_gaussians = run_anysplat_subprocess(
        images_dir=frames_dir,
        output_ply=output_ply,
        max_views=max_views,
        resolution=resolution,
        chunked=chunked,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    duration = time.time() - start
    logger.info(f"  AnySplat done: {n_gaussians:,} gaussians in {duration:.1f}s")

    # Save baseline copy
    backup = project_dir / "scene_baseline.ply"
    shutil.copy2(str(output_ply), str(backup))

    return duration, n_gaussians


def evaluate_ply(
    trainer, test_indices: list[int],
    all_cameras: list[dict], all_images: list[torch.Tensor],
    compute_lpips: bool = False,
) -> dict:
    """Evaluate a trainer's current state on held-out views."""
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
            logger.warning("LPIPS unavailable")

    with torch.no_grad():
        for idx in test_indices:
            cam = all_cameras[idx]
            gt = all_images[idx]
            W, H = cam["width"], cam["height"]

            viewmat = torch.tensor(cam["viewmat"], device=device).unsqueeze(0)
            K = torch.tensor(cam["K"], device=device).unsqueeze(0)

            result = trainer.rasterizer.rasterize(
                trainer.params, viewmat, K, W, H,
                trainer.config.sh_degree,
            )
            rendered = result.image.clamp(0, 1)

            psnrs.append(compute_psnr(rendered, gt))
            ssims.append(compute_ssim(rendered, gt))
            if lpips_net is not None:
                lpipss.append(_compute_lpips(rendered, gt, lpips_net))

    if lpips_net is not None:
        del lpips_net
        torch.cuda.empty_cache()

    out = {"mean_psnr": float(np.mean(psnrs)), "mean_ssim": float(np.mean(ssims))}
    if lpipss:
        out["mean_lpips"] = float(np.mean(lpipss))
    return out


def load_ply_params(ply_path: Path) -> dict:
    """Load all Gaussian parameters from a PLY file."""
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    return {
        "means": np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32),
        "scales": np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float32),
        "quats": np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1).astype(np.float32),
        "opacities": np.array(vertex["opacity"]).astype(np.float32),
        "sh0": np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32),
        "n_gaussians": len(vertex.data),
    }


def create_trainer_from_ply(
    ply_path: Path, config_overrides: dict,
    all_cameras: list[dict], all_images: list[torch.Tensor],
    train_indices: list[int], test_indices: list[int],
):
    """Create a GaussianTrainer initialized from an existing PLY."""
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig

    params = load_ply_params(ply_path)

    config = TrainerConfig(
        iterations=config_overrides.get("iterations", 0),
        sh_degree=config_overrides.get("sh_degree", 1),
        mode="3dgs",
        depth_reg_weight=config_overrides.get("depth_reg_weight", 0.05),
        opacity_reg_weight=config_overrides.get("opacity_reg_weight", 0.01),
        scale_reg_weight=config_overrides.get("scale_reg_weight", 0.01),
        prune_opa=0.005,
        densify_until_pct=0.5,
        means_lr=config_overrides.get("means_lr", 1.6e-4),
        means_lr_final=config_overrides.get("means_lr_final", 1.6e-6),
        scales_lr=config_overrides.get("scales_lr", 5e-3),
        quats_lr=config_overrides.get("quats_lr", 1e-3),
        opacities_lr=config_overrides.get("opacities_lr", 5e-2),
        sh0_lr=config_overrides.get("sh0_lr", 2.5e-3),
        shN_lr=config_overrides.get("shN_lr", 2.5e-3),
        densify_enabled=config_overrides.get("densify_enabled", True),
        reset_opacity=config_overrides.get("reset_opacity", True),
        freeze_positions_pct=config_overrides.get("freeze_positions_pct", 0.0),
        grow_grad2d=config_overrides.get("grow_grad2d", 0.0002),
    )

    trainer = GaussianTrainer(config)
    trainer.cameras_data = list(all_cameras)
    trainer.gt_images = list(all_images)
    trainer._all_cameras = all_cameras
    trainer._all_images = all_images
    trainer._eval_views = test_indices

    trainer.init_params_from_existing(
        params["means"], params["scales"], params["quats"],
        params["opacities"], params["sh0"],
    )
    trainer.init_strategy()
    trainer.set_training_views(train_indices)

    return trainer, params["n_gaussians"]


def evaluate_baseline(
    ply_path: Path,
    all_cameras: list[dict], all_images: list[torch.Tensor],
    test_indices: list[int], compute_lpips: bool = False,
) -> StageResult:
    """Evaluate AnySplat output before refinement."""
    from pipeline.train_gaussians import GaussianTrainer, TrainerConfig

    params = load_ply_params(ply_path)

    config = TrainerConfig(iterations=0, sh_degree=0, densify_enabled=False)
    trainer = GaussianTrainer(config)
    trainer.cameras_data = all_cameras
    trainer.gt_images = all_images
    trainer.init_params_from_existing(
        params["means"], params["scales"], params["quats"],
        params["opacities"], params["sh0"],
    )

    result = evaluate_ply(trainer, test_indices, all_cameras, all_images, compute_lpips)
    trainer.cleanup()

    return StageResult(
        stage="baseline",
        psnr=result["mean_psnr"],
        ssim=result["mean_ssim"],
        lpips=result.get("mean_lpips"),
        n_gaussians=params["n_gaussians"],
    )


def run_refinement(
    project_dir: Path, preset_name: str,
    all_cameras: list[dict], all_images: list[torch.Tensor],
    train_indices: list[int], test_indices: list[int],
    compute_lpips: bool = False,
) -> StageResult:
    """Run refinement with a preset and evaluate."""
    from app.routes.postprocess import REFINE_PRESETS
    from pipeline.train_gaussians import TrainerCallbacks

    ply_path = project_dir / "scene.ply"
    preset_def = REFINE_PRESETS[preset_name]
    preset_config = preset_def["config"]

    trainer, n_orig = create_trainer_from_ply(
        ply_path, preset_config,
        all_cameras, all_images,
        train_indices, test_indices,
    )

    # Track metrics
    train_metrics = []
    eval_metrics = []

    callbacks = TrainerCallbacks(
        metrics=lambda m: train_metrics.append(m),
        evaluation=lambda e: eval_metrics.append(e),
        metrics_every=50,
        eval_every=500,
    )

    output_path = project_dir / f"refined_{preset_name}.ply"

    logger.info(f"  Refining with '{preset_name}' ({preset_config['iterations']} iters)...")
    start = time.time()
    n_final = trainer.train(callbacks, output_path)
    trainer.export(output_path)
    duration = time.time() - start

    # Evaluate
    result = evaluate_ply(trainer, test_indices, all_cameras, all_images, compute_lpips)

    extra = {
        "iterations_actual": trainer.current_step,
        "loss_start": train_metrics[0]["losses"]["total"] if train_metrics else None,
        "loss_end": train_metrics[-1]["losses"]["total"] if train_metrics else None,
        "preset_config": preset_config,
    }
    if eval_metrics:
        extra["eval_psnr_curve"] = [
            {"step": e["step"], "psnr": e["mean_psnr"]} for e in eval_metrics
        ]

    trainer.cleanup()

    return StageResult(
        stage=f"refine_{preset_name}",
        psnr=result["mean_psnr"],
        ssim=result["mean_ssim"],
        lpips=result.get("mean_lpips"),
        n_gaussians=n_final,
        duration_sec=duration,
        extra=extra,
    )


# ---------- Per-scene pipeline ----------

def run_scene_pipeline(
    scene_name: str, max_views: int, resolution: int, downscale: int,
    presets: list[str], compute_lpips: bool, skip_anysplat: bool,
    chunked: bool = False, chunk_size: int = 32, chunk_overlap: int = 8,
) -> dict:
    """Run the full pipeline for one scene and create/update its project."""
    scene_dir = BENCHMARKS_DIR / scene_name
    if not scene_dir.exists():
        logger.error(f"Scene not found: {scene_dir}")
        return {}

    project_name = f"Benchmark: {scene_name.title()}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Scene: {scene_name} ({max_views} views @ {resolution}px)")
    logger.info(f"{'='*60}")

    # Reuse existing project or create new one
    project_id = db_find_project_by_name(project_name)
    if project_id:
        logger.info(f"  Reusing existing project {project_id}")
        db_update_project(project_id, status="processing", reconstruction_mode="anysplat")
    else:
        project_id = db_create_project(project_name)
        logger.info(f"  Created project {project_id}")

    project_dir = PROJECTS_DIR / project_id

    # 1. Copy benchmark frames
    logger.info("Step 1: Preparing frames...")
    image_paths = select_benchmark_images(scene_dir, downscale, max_views)
    copy_frames_to_project(image_paths, project_dir)

    # Also copy images to sources/ so they show in the UI upload list
    sources_dir = project_dir / "sources"
    sources_dir.mkdir(exist_ok=True)
    # Create a manifest file indicating these are benchmark images
    manifest = {
        "source": f"mipnerf360/{scene_name}",
        "downscale": downscale,
        "n_images": len(image_paths),
        "original_paths": [str(p) for p in image_paths],
    }
    with open(sources_dir / "benchmark_info.json", "w") as f:
        json.dump(manifest, f, indent=2)

    db_update_project(
        project_id,
        video_filename=",".join(f"frame_{i:05d}.jpg" for i in range(len(image_paths))),
    )

    # 2. Run AnySplat
    ply_path = project_dir / "scene.ply"
    baseline_path = project_dir / "scene_baseline.ply"

    if skip_anysplat and ply_path.exists():
        logger.info("Step 2: Skipping AnySplat (using existing scene.ply)")
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        anysplat_n = len(plydata["vertex"].data)
        anysplat_duration = 0.0
        if not baseline_path.exists():
            shutil.copy2(str(ply_path), str(baseline_path))
    else:
        logger.info("Step 2: Running AnySplat...")
        anysplat_duration, anysplat_n = run_anysplat(
            project_dir, max_views, resolution,
            chunked=chunked, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

    db_update_project(project_id, gaussian_count=anysplat_n, reconstruction_mode="anysplat")

    # 3. Load cameras and images for evaluation
    logger.info("Step 3: Loading cameras and images...")
    all_cameras, all_images = load_cameras_and_images(project_dir)
    n_views = len(all_cameras)

    from app.routes.postprocess import _split_train_test
    train_indices, test_indices = _split_train_test(n_views, test_every=4)
    logger.info(f"  {n_views} views: {len(train_indices)} train, {len(test_indices)} test")

    if not test_indices:
        test_indices = list(range(n_views))

    # 4. Evaluate baseline
    logger.info("Step 4: Evaluating baseline (AnySplat output)...")
    baseline = evaluate_baseline(ply_path, all_cameras, all_images, test_indices, compute_lpips)
    baseline.duration_sec = anysplat_duration
    logger.info(f"  Baseline: PSNR={baseline.psnr:.2f} dB, SSIM={baseline.ssim:.4f}")

    # 4b. Geometric quality measures
    cameras_path = project_dir / "cameras.json"
    geo_quality = None
    if cameras_path.exists():
        logger.info("Step 4b: Computing geometric quality measures...")
        try:
            from pipeline.geometric_quality import compute_geometric_quality
            geo_quality = compute_geometric_quality(ply_path, cameras_path)
            logger.info(
                f"  Health score: {geo_quality['health_score']}/100 "
                f"(depth={geo_quality['depth_consistency']['pct_consistent']:.2f}, "
                f"edges={geo_quality['edge_alignment']['mean_edge_alignment']:.2f}, "
                f"alpha={geo_quality['alpha_coverage']['mean_alpha']:.2f})"
            )
            # Free GPU memory before refinement
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"  Geometric quality failed: {e}")
    else:
        logger.info("Step 4b: Skipping geometric quality (no cameras.json)")

    # 5. Run each refinement preset
    all_stages = [baseline]
    best_preset = None
    best_psnr = baseline.psnr

    for preset in presets:
        logger.info(f"\nStep 5: Refinement — {preset}")

        # Restore baseline PLY for each preset
        shutil.copy2(str(baseline_path), str(ply_path))

        stage = run_refinement(
            project_dir, preset,
            all_cameras, all_images,
            train_indices, test_indices,
            compute_lpips,
        )
        all_stages.append(stage)

        delta = stage.psnr - baseline.psnr
        logger.info(f"  {preset}: PSNR={stage.psnr:.2f} ({delta:+.2f}), SSIM={stage.ssim:.4f}")

        if stage.psnr > best_psnr:
            best_psnr = stage.psnr
            best_preset = preset

    # 6. Set scene.ply to best refinement result (or keep baseline if none improved)
    if best_preset:
        best_ply = project_dir / f"refined_{best_preset}.ply"
        if best_ply.exists():
            shutil.copy2(str(best_ply), str(ply_path))
            logger.info(f"  Set scene.ply to best result: {best_preset} ({best_psnr:.2f} dB)")
            # Update gaussian count from refined version
            from plyfile import PlyData
            plydata = PlyData.read(str(ply_path))
            db_update_project(project_id, gaussian_count=len(plydata["vertex"].data))
    else:
        # Restore baseline
        shutil.copy2(str(baseline_path), str(ply_path))
        logger.info("  No improvement from refinement, keeping AnySplat baseline")

    # 7. Mark project as ready
    db_update_project(project_id, status="ready")
    db_create_job(project_id, "reconstruct")

    # 8. Save results JSON in project dir
    scene_results = {
        "scene": scene_name,
        "project_id": project_id,
        "max_views": max_views,
        "resolution": resolution,
        "downscale": downscale,
        "chunked": chunked,
        "best_preset": best_preset,
        "geometric_quality": geo_quality,
        "stages": [],
    }
    for s in all_stages:
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
            # Filter out non-serializable items
            d["extra"] = {k: v for k, v in s.extra.items() if k != "preset_config"}
        scene_results["stages"].append(d)

    with open(project_dir / "benchmark_results.json", "w") as f:
        json.dump(scene_results, f, indent=2)

    # Free GPU memory
    del all_cameras, all_images
    torch.cuda.empty_cache()

    return scene_results


# ---------- Summary display ----------

def print_summary(all_results: list[dict]):
    """Print a summary table across all scenes."""
    print()
    print("=" * 100)
    print(f"{'INDOOR BENCHMARK RESULTS':^100}")
    print("=" * 100)

    header = (
        f"{'Scene':<12} {'Stage':<20} {'PSNR':>8} {'dPSNR':>8} "
        f"{'SSIM':>8} {'#Gauss':>10} {'Time':>8}"
    )
    print(header)
    print("-" * 100)

    for res in all_results:
        scene = res["scene"]
        chunked_tag = " [C]" if res.get("chunked") else ""
        stages = res["stages"]
        baseline_psnr = stages[0]["psnr"] if stages else 0

        for s in stages:
            delta = s["psnr"] - baseline_psnr
            prefix = "+" if delta >= 0 else ""
            delta_str = f"{prefix}{delta:.2f}" if s["stage"] != "baseline" else "---"
            print(
                f"{scene + chunked_tag:<12} {s['stage']:<20} {s['psnr']:>8.2f} {delta_str:>8} "
                f"{s['ssim']:>8.4f} {s['n_gaussians']:>10,} {s['duration_sec']:>7.1f}s"
            )
        print()

    # Geometric quality summary
    has_geo = any(res.get("geometric_quality") for res in all_results)
    if has_geo:
        print(f"\n{'GEOMETRIC QUALITY':^100}")
        print("-" * 100)
        header_geo = (
            f"{'Scene':<16} {'Health':>8} {'Depth':>8} {'Edges':>8} "
            f"{'Alpha':>8} {'Clean':>8} {'OpaHP':>8}"
        )
        print(header_geo)
        print("-" * 100)
        for res in all_results:
            geo = res.get("geometric_quality")
            if not geo:
                continue
            scene = res["scene"]
            chunked_tag = " [C]" if res.get("chunked") else ""
            hc = geo["health_components"]
            print(
                f"{scene + chunked_tag:<16} {geo['health_score']:>8.1f} "
                f"{hc['geometry']:>8.3f} {hc['edge_alignment']:>8.3f} "
                f"{hc['coverage']:>8.3f} {hc['cleanliness']:>8.3f} "
                f"{hc['opacity_health']:>8.3f}"
            )
        print()

    # Summary
    print("-" * 100)
    for res in all_results:
        scene = res["scene"]
        pid = res["project_id"]
        best = res.get("best_preset", "none")
        mode = "chunked" if res.get("chunked") else "single-pass"
        print(f"  {scene}: project_id={pid}, best_preset={best}, mode={mode}")

    print()
    print("Projects are now viewable in the UI at http://localhost:5173")
    print()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Run indoor benchmark pipeline")
    parser.add_argument("--scenes", nargs="+", default=INDOOR_SCENES,
                        help=f"Scenes to test (default: {INDOOR_SCENES})")
    parser.add_argument("--views", type=int, default=None, help="Max views for AnySplat")
    parser.add_argument("--resolution", type=int, default=None, help="Override resolution")
    parser.add_argument("--downscale", type=int, default=None, help="Image downscale (1,2,4,8)")
    parser.add_argument("--presets", nargs="+", default=None, help="Refinement presets")
    parser.add_argument("--quick", action="store_true", help="Quick: 16 views, conservative only")
    parser.add_argument("--full", action="store_true", help="Full: 32 views, all presets + LPIPS")
    parser.add_argument("--skip-anysplat", action="store_true", help="Reuse existing scene.ply")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS")
    parser.add_argument("--chunked", action="store_true",
                        help="Use chunked pipeline with all available images")
    parser.add_argument("--chunk-size", type=int, default=32, help="Views per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=8, help="Overlap between chunks")
    args = parser.parse_args()

    # Defaults based on mode
    if args.quick:
        views = args.views or 16
        resolution = args.resolution or 224
        downscale = args.downscale or 8
        presets = args.presets or ["conservative"]
        compute_lpips = False
    elif args.full:
        views = args.views or 32
        resolution = args.resolution or 448
        downscale = args.downscale or 4
        presets = args.presets or ["conservative", "balanced", "aggressive"]
        compute_lpips = True
    else:
        views = args.views or 32
        resolution = args.resolution or 448
        downscale = args.downscale or 8
        presets = args.presets or ["balanced"]
        compute_lpips = args.lpips

    scenes = args.scenes

    # Verify benchmark data exists
    for scene in scenes:
        if not (BENCHMARKS_DIR / scene).exists():
            available = [d.name for d in BENCHMARKS_DIR.iterdir() if d.is_dir()]
            logger.error(f"Scene '{scene}' not found. Available: {available}")
            sys.exit(1)

    logger.info(f"Indoor benchmark: scenes={scenes}, views={views}, res={resolution}, "
                f"downscale={downscale}, presets={presets}")

    # Run each scene
    all_results = []
    total_start = time.time()

    for scene in scenes:
        result = run_scene_pipeline(
            scene, views, resolution, downscale,
            presets, compute_lpips, args.skip_anysplat,
            chunked=args.chunked,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if result:
            all_results.append(result)

    total_duration = time.time() - total_start

    # Save combined results
    combined_path = PROJECTS_DIR / "indoor_benchmark_results.json"
    with open(combined_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_sec": round(total_duration, 1),
            "config": {
                "views": views,
                "resolution": resolution,
                "downscale": downscale,
                "presets": presets,
                "chunked": args.chunked,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
            },
            "scenes": all_results,
        }, f, indent=2)

    print_summary(all_results)
    logger.info(f"Total time: {total_duration:.0f}s ({total_duration/60:.1f} min)")
    logger.info(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
