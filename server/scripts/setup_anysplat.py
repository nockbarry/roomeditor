#!/usr/bin/env python3
"""Set up AnySplat for feed-forward Gaussian Splatting from unposed images.

AnySplat (SIGGRAPH Asia 2025) is a feed-forward model that produces
3D Gaussian Splats from arbitrary unposed images in a single forward pass.
No COLMAP or MASt3R needed — it predicts both poses and Gaussians directly.

Trained on ARKitScenes (4,406 indoor scenes) and ScanNet++ (935 indoor scenes),
making it ideal for room-scale indoor reconstruction from phone video.

Requirements:
    - Python 3.10+
    - PyTorch 2.2.0+ with CUDA
    - ~6GB disk for model weights
    - ~16GB VRAM for inference (adjustable via batch_size/n_views)

Usage:
    # Install AnySplat:
    python scripts/setup_anysplat.py

    # Run on a folder of images:
    python scripts/setup_anysplat.py --test /path/to/images

    # Check installation:
    python scripts/setup_anysplat.py --check
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ANYSPLAT_DIR = Path("/home/nock/projects/anysplat")
ANYSPLAT_REPO = "https://github.com/OpenRobotLab/AnySplat.git"


def check_installation() -> bool:
    """Check if AnySplat is properly installed."""
    if not ANYSPLAT_DIR.exists():
        logger.info("AnySplat not found at %s", ANYSPLAT_DIR)
        return False

    # Check for key files
    required = [
        ANYSPLAT_DIR / "demo_gradio.py",
        ANYSPLAT_DIR / "src" / "model" / "model" / "anysplat.py",
        ANYSPLAT_DIR / "requirements.txt",
    ]
    for f in required:
        if not f.exists():
            logger.info("Missing: %s", f)
            return False

    # Try importing
    sys.path.insert(0, str(ANYSPLAT_DIR))
    try:
        import torch
        logger.info("PyTorch: %s, CUDA: %s", torch.__version__, torch.cuda.is_available())
    except ImportError:
        logger.error("PyTorch not available")
        return False

    logger.info("AnySplat installation looks good at %s", ANYSPLAT_DIR)
    return True


def install():
    """Clone and install AnySplat."""
    if ANYSPLAT_DIR.exists():
        logger.info("AnySplat directory already exists at %s", ANYSPLAT_DIR)
        logger.info("Delete it first to re-install: rm -rf %s", ANYSPLAT_DIR)
        return

    logger.info("Cloning AnySplat...")
    subprocess.run(
        ["git", "clone", ANYSPLAT_REPO, str(ANYSPLAT_DIR)],
        check=True,
    )

    logger.info("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ANYSPLAT_DIR / "requirements.txt")],
        check=True,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("AnySplat installed successfully!")
    logger.info("")
    logger.info("Pretrained weights will download automatically on first use.")
    logger.info("Model: ~886M parameters, ~6GB download")
    logger.info("")
    logger.info("Quick test:")
    logger.info("  python scripts/setup_anysplat.py --check")
    logger.info("")
    logger.info("Run Gradio demo:")
    logger.info("  cd %s && python demo_gradio.py", ANYSPLAT_DIR)
    logger.info("")
    logger.info("Run on images:")
    logger.info("  python scripts/setup_anysplat.py --test /path/to/images/")
    logger.info("=" * 60)


def run_test(images_dir: str):
    """Run AnySplat on a folder of images for quick evaluation."""
    import torch

    images_path = Path(images_dir)
    if not images_path.exists():
        logger.error("Images directory not found: %s", images_path)
        sys.exit(1)

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        f for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    if not image_files:
        logger.error("No images found in %s", images_path)
        sys.exit(1)

    logger.info("Found %d images in %s", len(image_files), images_path)

    # Limit views for 16GB VRAM
    max_views = 16
    if len(image_files) > max_views:
        step = max(1, len(image_files) // max_views)
        image_files = image_files[::step][:max_views]
        logger.info("Subsampled to %d views for VRAM budget", len(image_files))

    sys.path.insert(0, str(ANYSPLAT_DIR))

    logger.info("Loading AnySplat model...")
    from src.model.model.anysplat import AnySplat

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnySplat.from_pretrained("anysplat_ckpt_v1").to(device).eval()

    logger.info("Preprocessing images...")
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    images = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        images.append(transform(img))

    batch = torch.stack(images).unsqueeze(0).to(device)  # [1, N, 3, 448, 448]
    logger.info("Input tensor: %s", batch.shape)

    logger.info("Running inference...")
    from time import time
    t0 = time()

    with torch.no_grad():
        output = model(batch)

    elapsed = time() - t0
    logger.info("Inference complete in %.1fs", elapsed)

    # Extract results
    if hasattr(output, "gaussians"):
        gaussians = output.gaussians
        logger.info("Output Gaussians: %s", gaussians.shape if hasattr(gaussians, "shape") else type(gaussians))

    if hasattr(output, "cameras"):
        cameras = output.cameras
        logger.info("Predicted cameras: %s", type(cameras))

    # Save PLY if possible
    output_ply = images_path.parent / "anysplat_output.ply"
    try:
        if hasattr(output, "export_ply"):
            output.export_ply(str(output_ply))
            logger.info("Saved PLY to %s", output_ply)
        elif hasattr(output, "save"):
            output.save(str(output_ply))
            logger.info("Saved to %s", output_ply)
        else:
            logger.info("No direct export method — check AnySplat docs for PLY export")
    except Exception as e:
        logger.warning("Could not save PLY: %s", e)

    # Cleanup
    del model, output, batch
    torch.cuda.empty_cache()

    logger.info("")
    logger.info("Summary:")
    logger.info("  Views: %d", len(image_files))
    logger.info("  Inference time: %.1fs", elapsed)
    logger.info("  VRAM peak: %.1f GB", torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0)


def main():
    parser = argparse.ArgumentParser(description="Set up AnySplat")
    parser.add_argument("--check", action="store_true", help="Check installation")
    parser.add_argument("--test", type=str, help="Run on a folder of images")
    args = parser.parse_args()

    if args.check:
        ok = check_installation()
        sys.exit(0 if ok else 1)
    elif args.test:
        if not check_installation():
            logger.error("AnySplat not installed. Run without --check/--test to install.")
            sys.exit(1)
        run_test(args.test)
    else:
        install()


if __name__ == "__main__":
    main()
