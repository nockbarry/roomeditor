"""AI object generation pipeline (TripoSR / LGM).

This module will be implemented in Phase 3. Provides stubs for now.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def generate_from_image(
    image_path: Path, output_ply_path: Path
) -> int:
    """Generate a 3D Gaussian splat from an image using TripoSR.

    Returns the number of Gaussians generated.
    Phase 3 implementation.
    """
    raise NotImplementedError("Image-to-3D generation will be implemented in Phase 3")


async def generate_from_text(
    prompt: str, output_ply_path: Path
) -> int:
    """Generate a 3D Gaussian splat from a text prompt using LGM.

    Returns the number of Gaussians generated.
    Phase 3 implementation.
    """
    raise NotImplementedError("Text-to-3D generation will be implemented in Phase 3")
