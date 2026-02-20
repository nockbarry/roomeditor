#!/usr/bin/env python3
"""Download benchmark datasets for evaluation.

Currently supports:
- Mip-NeRF 360 (360_v2) — ~4.5GB compressed

Usage:
    python scripts/download_benchmarks.py [--dataset mipnerf360] [--output-dir data/benchmarks]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATASETS = {
    "mipnerf360": {
        "url": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "extract_dir": "mipnerf360",
        "description": "Mip-NeRF 360 dataset (9 scenes, ~4.5GB)",
    },
}

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "benchmarks"


def download_dataset(dataset: str, output_dir: Path):
    """Download and extract a benchmark dataset."""
    if dataset not in DATASETS:
        logger.error(f"Unknown dataset: {dataset}")
        logger.error(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    info = DATASETS[dataset]
    target_dir = output_dir / info["extract_dir"]

    if target_dir.exists():
        n_scenes = len([d for d in target_dir.iterdir() if d.is_dir()])
        if n_scenes > 0:
            logger.info(
                f"Dataset already exists at {target_dir} ({n_scenes} scenes). "
                f"Delete it to re-download."
            )
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{dataset}.zip"

    logger.info(f"Downloading {info['description']}...")
    logger.info(f"  URL: {info['url']}")
    logger.info(f"  Target: {target_dir}")

    # Download with wget (supports resume)
    try:
        subprocess.run(
            ["wget", "-c", "-O", str(zip_path), info["url"]],
            check=True,
        )
    except FileNotFoundError:
        # Fall back to curl
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(zip_path), info["url"]],
            check=True,
        )

    logger.info("Extracting...")
    # Extract — 360_v2.zip contains scene dirs at top level
    temp_dir = output_dir / "_extract_tmp"
    temp_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["unzip", "-o", "-q", str(zip_path), "-d", str(temp_dir)],
        check=True,
    )

    # Move extracted content to target dir
    target_dir.mkdir(exist_ok=True)
    for item in temp_dir.iterdir():
        if item.is_dir():
            dest = target_dir / item.name
            if dest.exists():
                import shutil
                shutil.rmtree(dest)
            item.rename(dest)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    n_scenes = len([d for d in target_dir.iterdir() if d.is_dir()])
    logger.info(f"Done! {n_scenes} scenes extracted to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        default="mipnerf360",
        choices=list(DATASETS.keys()),
        help="Dataset to download (default: mipnerf360)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    args = parser.parse_args()

    if args.list:
        for name, info in DATASETS.items():
            print(f"  {name}: {info['description']}")
        return

    download_dataset(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
