"""Benchmarks comparison API.

Scans data/benchmarks/results/ for local run results and returns
them alongside published baselines from 3DGS/2DGS papers.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])

RESULTS_DIR = settings.data_dir / "benchmarks" / "results"

# Published baselines (Mip-NeRF 360, images_4, from papers)
PUBLISHED_BASELINES = {
    "mipnerf360": {
        "3dgs": {
            "kitchen": {"psnr": 31.14, "ssim": 0.926, "lpips": 0.155},
            "room": {"psnr": 31.43, "ssim": 0.918, "lpips": 0.184},
            "counter": {"psnr": 29.28, "ssim": 0.913, "lpips": 0.180},
            "bonsai": {"psnr": 32.32, "ssim": 0.948, "lpips": 0.135},
            "bicycle": {"psnr": 25.15, "ssim": 0.765, "lpips": 0.246},
            "garden": {"psnr": 27.32, "ssim": 0.858, "lpips": 0.137},
            "stump": {"psnr": 26.55, "ssim": 0.770, "lpips": 0.261},
            "flowers": {"psnr": 21.52, "ssim": 0.605, "lpips": 0.336},
            "treehill": {"psnr": 22.49, "ssim": 0.631, "lpips": 0.339},
        },
        "2dgs": {
            "kitchen": {"psnr": 31.35, "ssim": 0.929, "lpips": 0.145},
            "room": {"psnr": 31.32, "ssim": 0.919, "lpips": 0.174},
            "counter": {"psnr": 29.16, "ssim": 0.912, "lpips": 0.172},
            "bonsai": {"psnr": 32.67, "ssim": 0.949, "lpips": 0.127},
            "bicycle": {"psnr": 25.19, "ssim": 0.758, "lpips": 0.247},
            "garden": {"psnr": 27.13, "ssim": 0.852, "lpips": 0.142},
            "stump": {"psnr": 26.61, "ssim": 0.766, "lpips": 0.262},
            "flowers": {"psnr": 21.30, "ssim": 0.587, "lpips": 0.346},
            "treehill": {"psnr": 22.62, "ssim": 0.628, "lpips": 0.341},
        },
    }
}


def _collect_local_results() -> list[dict]:
    """Scan results directory for results.json files."""
    results = []
    if not RESULTS_DIR.exists():
        return results

    for results_json in sorted(RESULTS_DIR.glob("**/results.json")):
        try:
            data = json.loads(results_json.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", results_json, e)
            continue

        method = data.get("method", "unknown")
        dataset = data.get("dataset", "unknown")
        scene = data.get("scene", "unknown")

        # Resolve render image URLs
        renders_dir = results_json.parent / "renders"
        render_urls: list[str] = []
        if renders_dir.is_dir():
            for img in sorted(renders_dir.iterdir()):
                if img.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    render_urls.append(
                        f"/data/benchmarks/results/{method}/{dataset}/{scene}/renders/{img.name}"
                    )

        results.append({
            "method": method,
            "dataset": dataset,
            "scene": scene,
            "metrics": data.get("metrics", {}),
            "config": data.get("config"),
            "training_time_sec": data.get("training_time_sec"),
            "num_gaussians": data.get("num_gaussians"),
            "per_view": data.get("per_view", []),
            "render_urls": render_urls,
            "is_published": False,
        })

    return results


def _build_published_results() -> list[dict]:
    """Convert published baselines into the same result shape."""
    results = []
    for dataset, methods in PUBLISHED_BASELINES.items():
        for method_key, scenes in methods.items():
            for scene, metrics in scenes.items():
                results.append({
                    "method": f"published-{method_key}",
                    "dataset": dataset,
                    "scene": scene,
                    "metrics": {
                        "mean_psnr": metrics["psnr"],
                        "mean_ssim": metrics["ssim"],
                        "mean_lpips": metrics["lpips"],
                    },
                    "config": None,
                    "training_time_sec": None,
                    "num_gaussians": None,
                    "per_view": [],
                    "render_urls": [],
                    "is_published": True,
                })
    return results


@router.get("")
async def get_benchmarks():
    local = _collect_local_results()
    published = _build_published_results()

    all_results = local + published

    datasets: set[str] = set()
    methods: set[str] = set()
    scenes_by_dataset: dict[str, set[str]] = {}

    for r in all_results:
        ds = r["dataset"]
        datasets.add(ds)
        methods.add(r["method"])
        scenes_by_dataset.setdefault(ds, set()).add(r["scene"])

    return {
        "results": local,
        "published": published,
        "datasets": sorted(datasets),
        "methods": sorted(methods),
        "scenes_by_dataset": {
            ds: sorted(scenes) for ds, scenes in scenes_by_dataset.items()
        },
    }
