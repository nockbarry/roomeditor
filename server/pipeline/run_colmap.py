import logging
from pathlib import Path
from typing import Callable

import pycolmap

logger = logging.getLogger(__name__)


def run_colmap(
    images_dir: Path,
    output_dir: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict]:
    """Run COLMAP SfM pipeline using pycolmap Python API.

    Steps: feature_extractor -> sequential_matcher -> mapper
    Returns (model_dir, metadata) where metadata includes reconstruction stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale database if exists
    if db_path.exists():
        db_path.unlink()

    if progress_callback:
        progress_callback(0.0)

    # Step 1: Feature extraction
    logger.info(f"COLMAP: extracting features from {images_dir}")
    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.sift.max_num_features = 8192
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="OPENCV",
        extraction_options=extraction_opts,
    )

    if progress_callback:
        progress_callback(0.3)

    # Step 2: Sequential matching (optimal for video — O(n) instead of O(n^2))
    logger.info("COLMAP: matching features (sequential)")
    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.guided_matching = True
    pairing_opts = pycolmap.SequentialPairingOptions()
    pairing_opts.overlap = 10
    pairing_opts.quadratic_overlap = True
    pairing_opts.loop_detection = False
    pycolmap.match_sequential(
        database_path=str(db_path),
        matching_options=matching_opts,
        pairing_options=pairing_opts,
    )

    if progress_callback:
        progress_callback(0.6)

    # Step 3: Sparse reconstruction (mapper)
    logger.info("COLMAP: running mapper (sparse reconstruction)")
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.ba_global_max_num_iterations = 50
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
        options=mapper_opts,
    )

    if not reconstructions:
        raise RuntimeError(
            "COLMAP produced no reconstruction. This usually means the video "
            "doesn't have enough feature overlap. Try filming more slowly with "
            "more overlap between views."
        )

    # Pick the largest reconstruction (most registered images)
    sorted_recons = sorted(
        reconstructions.items(),
        key=lambda kv: len(kv[1].images),
        reverse=True,
    )
    best_idx, best_recon = sorted_recons[0]
    num_images = len(best_recon.images)
    num_points = len(best_recon.points3D)

    if len(sorted_recons) > 1:
        other_counts = [len(r.images) for _, r in sorted_recons[1:]]
        logger.warning(
            f"COLMAP produced {len(sorted_recons)} separate reconstructions. "
            f"Using largest ({num_images} images). "
            f"Others had {other_counts} images respectively. "
            f"This may indicate multiple separate scenes in the input."
        )

    model_dir = sparse_dir / str(best_idx)
    if not model_dir.exists():
        model_dirs = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
        if model_dirs:
            model_dir = model_dirs[0]
        else:
            raise RuntimeError("COLMAP mapper produced no output directory")

    logger.info(
        f"COLMAP reconstruction complete: {num_images} images, "
        f"{num_points} 3D points in {model_dir}"
    )

    if progress_callback:
        progress_callback(1.0)

    meta = {
        "num_reconstructions": len(reconstructions),
        "registered_images": num_images,
        "total_points": num_points,
    }
    return model_dir, meta
