"""GPU memory and device management utilities."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_gpu_info() -> dict | None:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return {
                    "name": parts[0],
                    "memory_total": parts[1],
                    "memory_free": parts[2],
                }
    except FileNotFoundError:
        logger.warning("nvidia-smi not found")
    return None


def check_gpu_available() -> bool:
    """Check if a CUDA-capable GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return get_gpu_info() is not None
