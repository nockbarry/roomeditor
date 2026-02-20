#!/usr/bin/env python3
"""Compare evaluation results across methods.

Reads results.json files from the benchmarks results directory and
produces a comparison table. Extensible to any method that outputs
the standard results.json format.

Usage:
    python scripts/compare_methods.py
    python scripts/compare_methods.py --dataset mipnerf360 --indoor-only
    python scripts/compare_methods.py --results-dir data/benchmarks/results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.benchmark import MIPNERF360_INDOOR, MIPNERF360_OUTDOOR

DEFAULT_RESULTS_DIR = (
    Path(__file__).parent.parent / "data" / "benchmarks" / "results"
)

# Published baselines for reference (Mip-NeRF 360, images_4, from papers)
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


def collect_results(results_dir: Path) -> list[dict]:
    """Collect all results.json files from the results directory."""
    results = []
    for results_json in sorted(results_dir.glob("**/results.json")):
        try:
            with open(results_json) as f:
                data = json.load(f)
            data["_path"] = str(results_json)
            results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {results_json}: {e}", file=sys.stderr)
    return results


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds <= 0:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m{secs:02d}s"


def format_count(n: int) -> str:
    """Format Gaussian count with K/M suffix."""
    if n <= 0:
        return "--"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)


def print_comparison(
    results: list[dict],
    dataset: str = "mipnerf360",
    scenes: list[str] | None = None,
    include_published: bool = True,
):
    """Print a formatted comparison table."""
    # Group results by scene
    by_scene: dict[str, list[dict]] = {}
    for r in results:
        if r.get("dataset") != dataset:
            continue
        scene = r.get("scene", "unknown")
        if scenes and scene not in scenes:
            continue
        by_scene.setdefault(scene, []).append(r)

    if not by_scene and not include_published:
        print("No results found.")
        return

    # Collect all method names
    methods = set()
    for scene_results in by_scene.values():
        for r in scene_results:
            methods.add(r.get("method", "unknown"))

    # Add published baselines
    published = PUBLISHED_BASELINES.get(dataset, {})
    if include_published:
        for baseline_name in published:
            methods.add(f"published-{baseline_name}")

    # Get all scenes
    all_scenes = set(by_scene.keys())
    if include_published:
        for baseline in published.values():
            if scenes:
                all_scenes.update(s for s in baseline if s in scenes)
            else:
                all_scenes.update(baseline.keys())
    all_scenes = sorted(all_scenes)

    if not all_scenes:
        print("No scenes to compare.")
        return

    # Determine label
    if scenes and set(scenes) == set(MIPNERF360_INDOOR):
        label = f"{dataset} Indoor"
    elif scenes and set(scenes) == set(MIPNERF360_OUTDOOR):
        label = f"{dataset} Outdoor"
    else:
        label = dataset

    # Print table
    header = f"{label} Benchmark Comparison"
    print(f"\n{header}")
    print("=" * 78)
    print(
        f"{'Method':<20} {'Scene':<12} {'PSNR':>7} {'SSIM':>7} "
        f"{'LPIPS':>7} {'Time':>8} {'#GS':>7}"
    )
    print("-" * 78)

    # Aggregate per method for means
    method_metrics: dict[str, dict[str, list]] = {}

    for scene in all_scenes:
        scene_results = by_scene.get(scene, [])

        for r in scene_results:
            method = r.get("method", "unknown")
            m = r.get("metrics", {})
            t = r.get("training_time_sec", 0)
            n = r.get("num_gaussians", 0)
            print(
                f"{method:<20} {scene:<12} "
                f"{m.get('mean_psnr', 0):>7.2f} "
                f"{m.get('mean_ssim', 0):>7.4f} "
                f"{m.get('mean_lpips', 0):>7.4f} "
                f"{format_time(t):>8} "
                f"{format_count(n):>7}"
            )
            method_metrics.setdefault(method, {"psnr": [], "ssim": [], "lpips": []})
            method_metrics[method]["psnr"].append(m.get("mean_psnr", 0))
            method_metrics[method]["ssim"].append(m.get("mean_ssim", 0))
            method_metrics[method]["lpips"].append(m.get("mean_lpips", 0))

        # Published baselines for this scene
        if include_published:
            for baseline_name, baseline_scenes in published.items():
                if scene in baseline_scenes:
                    method = f"published-{baseline_name}"
                    m = baseline_scenes[scene]
                    print(
                        f"{method:<20} {scene:<12} "
                        f"{m['psnr']:>7.2f} "
                        f"{m['ssim']:>7.4f} "
                        f"{m['lpips']:>7.4f} "
                        f"{'--':>8} "
                        f"{'--':>7}"
                    )
                    method_metrics.setdefault(method, {"psnr": [], "ssim": [], "lpips": []})
                    method_metrics[method]["psnr"].append(m["psnr"])
                    method_metrics[method]["ssim"].append(m["ssim"])
                    method_metrics[method]["lpips"].append(m["lpips"])

    # Print means
    if method_metrics:
        print("-" * 78)
        print("Mean across scenes:")
        print("-" * 78)
        for method in sorted(method_metrics.keys()):
            mm = method_metrics[method]
            if mm["psnr"]:
                avg_psnr = sum(mm["psnr"]) / len(mm["psnr"])
                avg_ssim = sum(mm["ssim"]) / len(mm["ssim"])
                avg_lpips = sum(mm["lpips"]) / len(mm["lpips"])
                n_scenes = len(mm["psnr"])
                print(
                    f"{method:<20} {'(' + str(n_scenes) + ' scenes)':<12} "
                    f"{avg_psnr:>7.2f} "
                    f"{avg_ssim:>7.4f} "
                    f"{avg_lpips:>7.4f}"
                )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across methods"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument("--dataset", default="mipnerf360")
    parser.add_argument("--indoor-only", action="store_true")
    parser.add_argument("--outdoor-only", action="store_true")
    parser.add_argument("--no-published", action="store_true",
                        help="Don't show published baselines")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of table")
    args = parser.parse_args()

    results = collect_results(args.results_dir)

    if args.indoor_only:
        scenes = MIPNERF360_INDOOR
    elif args.outdoor_only:
        scenes = MIPNERF360_OUTDOOR
    else:
        scenes = None

    if args.json:
        filtered = [
            r for r in results
            if r.get("dataset") == args.dataset
            and (scenes is None or r.get("scene") in scenes)
        ]
        print(json.dumps(filtered, indent=2, default=str))
    else:
        print_comparison(
            results,
            dataset=args.dataset,
            scenes=scenes,
            include_published=not args.no_published,
        )


if __name__ == "__main__":
    main()
