#!/bin/bash
# Set up InstantSplat in a separate venv (conflicts with gsplat).
#
# Usage:
#   bash scripts/setup_instantsplat.sh
#
# This creates /home/nock/projects/instantsplat/ with its own venv.

set -euo pipefail

INSTALL_DIR="/home/nock/projects/instantsplat"
CUDA_HOME="$HOME/.local/cuda-12.8"

echo "=== Setting up InstantSplat ==="
echo "Install directory: $INSTALL_DIR"
echo "CUDA_HOME: $CUDA_HOME"

# Clone if not already present
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning InstantSplat..."
    cd /home/nock/projects
    git clone --recursive https://github.com/NVlabs/InstantSplat.git instantsplat
else
    echo "InstantSplat already cloned at $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# Create venv if not present
if [ ! -d ".venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

echo "Installing PyTorch with CUDA 12.8..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "Installing requirements..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "Installing submodules..."
for submod in submodules/simple-knn submodules/diff-gaussian-rasterization submodules/fused-ssim; do
    if [ -d "$submod" ]; then
        echo "  Installing $submod..."
        CUDA_HOME="$CUDA_HOME" pip install --no-build-isolation "$submod"
    else
        echo "  Skipping $submod (not found)"
    fi
done

# Download MASt3R checkpoint
mkdir -p checkpoints
CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ ! -f "$CKPT" ]; then
    echo "Downloading MASt3R checkpoint..."
    wget -c "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
        -O "$CKPT"
else
    echo "MASt3R checkpoint already present"
fi

echo ""
echo "=== InstantSplat setup complete ==="
echo "Venv: $INSTALL_DIR/.venv"
echo "Activate: source $INSTALL_DIR/.venv/bin/activate"
