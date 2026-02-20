"""Benchmark scene discovery and path resolution.

Supports Mip-NeRF 360 and other datasets in standard COLMAP format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard Mip-NeRF 360 scenes
MIPNERF360_INDOOR = ["kitchen", "room", "counter", "bonsai"]
MIPNERF360_OUTDOOR = ["bicycle", "garden", "stump", "flowers", "treehill"]
MIPNERF360_ALL = MIPNERF360_INDOOR + MIPNERF360_OUTDOOR

DEFAULT_BENCHMARKS_DIR = Path(__file__).parent.parent / "data" / "benchmarks"


@dataclass
class BenchmarkScene:
    """A benchmark scene with paths to COLMAP data and images."""

    name: str
    dataset: str
    colmap_dir: Path  # sparse/0/
    images_dir: Path  # images/ or images_2/ or images_4/
    downscale: int = 1

    @property
    def display_name(self) -> str:
        return f"{self.dataset}/{self.name}"


def get_scene(
    benchmarks_dir: Path,
    dataset: str,
    scene: str,
    downscale: int = 4,
) -> BenchmarkScene:
    """Get a specific benchmark scene.

    Args:
        benchmarks_dir: Root benchmarks directory.
        dataset: Dataset name (e.g., "mipnerf360").
        scene: Scene name (e.g., "kitchen").
        downscale: Image downscale factor (1, 2, or 4). Default 4.

    Returns:
        BenchmarkScene with resolved paths.

    Raises:
        FileNotFoundError: If scene directory doesn't exist.
    """
    scene_dir = benchmarks_dir / dataset / scene

    if not scene_dir.exists():
        raise FileNotFoundError(
            f"Benchmark scene not found: {scene_dir}\n"
            f"Run scripts/download_benchmarks.py to download the dataset."
        )

    # Find COLMAP sparse model
    colmap_dir = scene_dir / "sparse" / "0"
    if not colmap_dir.exists():
        # Some datasets put it directly in sparse/
        colmap_dir = scene_dir / "sparse"
    if not colmap_dir.exists():
        raise FileNotFoundError(
            f"COLMAP sparse model not found in {scene_dir}"
        )

    # Find images directory at requested downscale
    if downscale > 1:
        images_dir = scene_dir / f"images_{downscale}"
        if not images_dir.exists():
            logger.warning(
                f"images_{downscale}/ not found, falling back to images/"
            )
            images_dir = scene_dir / "images"
    else:
        images_dir = scene_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found in {scene_dir}")

    return BenchmarkScene(
        name=scene,
        dataset=dataset,
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        downscale=downscale,
    )


def discover_scenes(
    benchmarks_dir: Path | None = None,
) -> list[BenchmarkScene]:
    """Discover all available benchmark scenes.

    Args:
        benchmarks_dir: Root benchmarks directory. Uses default if None.

    Returns:
        List of available BenchmarkScene objects.
    """
    if benchmarks_dir is None:
        benchmarks_dir = DEFAULT_BENCHMARKS_DIR

    if not benchmarks_dir.exists():
        return []

    scenes = []
    for dataset_dir in sorted(benchmarks_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            # Check for COLMAP data
            sparse_dir = scene_dir / "sparse" / "0"
            if not sparse_dir.exists():
                sparse_dir = scene_dir / "sparse"
            if not sparse_dir.exists():
                continue

            # Check for images
            images_dir = scene_dir / "images"
            if not images_dir.exists():
                continue

            # Determine best available downscale
            downscale = 1
            for ds in [4, 2]:
                if (scene_dir / f"images_{ds}").exists():
                    downscale = ds
                    break

            scenes.append(
                BenchmarkScene(
                    name=scene_dir.name,
                    dataset=dataset,
                    colmap_dir=sparse_dir,
                    images_dir=scene_dir / (
                        f"images_{downscale}" if downscale > 1 else "images"
                    ),
                    downscale=downscale,
                )
            )

    return scenes
