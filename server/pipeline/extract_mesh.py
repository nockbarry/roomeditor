"""Mesh extraction from trained Gaussian Splatting models.

Uses TSDF fusion of rendered depth maps followed by marching cubes
to produce a triangle mesh suitable for physics, collision, and export.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TSDFVolume:
    """Truncated Signed Distance Function volume for depth fusion."""

    def __init__(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        voxel_size: float = 0.02,
        trunc_margin: float = 0.06,
    ):
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin

        # Compute volume dimensions
        dims = np.ceil((bounds_max - bounds_min) / voxel_size).astype(int)
        # Cap to reasonable size (512^3 max)
        max_dim = 512
        if np.any(dims > max_dim):
            scale = max_dim / dims.max()
            self.voxel_size = voxel_size / scale
            dims = np.ceil((bounds_max - bounds_min) / self.voxel_size).astype(int)
            logger.info(
                f"TSDF volume too large, adjusted voxel_size to "
                f"{self.voxel_size:.4f}"
            )

        self.dims = dims
        self.origin = bounds_min
        self.tsdf = np.ones(dims, dtype=np.float32)
        self.weight = np.zeros(dims, dtype=np.float32)
        self.color = np.zeros((*dims, 3), dtype=np.float32)

        logger.info(
            f"TSDF volume: {dims[0]}x{dims[1]}x{dims[2]} "
            f"(voxel_size={self.voxel_size:.4f}m)"
        )

    def integrate(
        self,
        depth: np.ndarray,
        color: np.ndarray,
        K: np.ndarray,
        viewmat: np.ndarray,
    ):
        """Integrate a depth map into the TSDF volume.

        Args:
            depth: (H, W) depth map in meters
            color: (H, W, 3) RGB image [0, 1]
            K: (3, 3) camera intrinsics
            viewmat: (4, 4) world-to-camera transform
        """
        H, W = depth.shape

        # Generate voxel world coordinates
        x = np.arange(self.dims[0]) * self.voxel_size + self.origin[0]
        y = np.arange(self.dims[1]) * self.voxel_size + self.origin[1]
        z = np.arange(self.dims[2]) * self.voxel_size + self.origin[2]
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        world_pts = np.stack([xv, yv, zv, np.ones_like(xv)], axis=-1)  # (X, Y, Z, 4)

        # Transform to camera coordinates
        cam_pts = (viewmat @ world_pts.reshape(-1, 4).T).T  # (N, 4)
        cam_pts = cam_pts.reshape(*self.dims, 4)

        # Project to image
        cam_z = cam_pts[..., 2]
        valid_z = cam_z > 0.01

        cam_z_safe = np.where(valid_z, cam_z, 1.0)  # Avoid divide-by-zero
        px = (K[0, 0] * cam_pts[..., 0] / cam_z_safe + K[0, 2]).astype(int)
        py = (K[1, 1] * cam_pts[..., 1] / cam_z_safe + K[1, 2]).astype(int)

        valid = valid_z & (px >= 0) & (px < W) & (py >= 0) & (py < H)

        # Look up depth values
        px_safe = np.clip(px, 0, W - 1)
        py_safe = np.clip(py, 0, H - 1)
        obs_depth = depth[py_safe, px_safe]
        obs_color = color[py_safe, px_safe]

        # Compute TSDF value
        sdf = obs_depth - cam_z
        valid &= obs_depth > 0
        valid &= sdf > -self.trunc_margin

        tsdf_val = np.clip(sdf / self.trunc_margin, -1.0, 1.0)

        # Update TSDF with running weighted average
        w_old = self.weight[valid]
        w_new = 1.0
        w_sum = w_old + w_new

        self.tsdf[valid] = (
            self.tsdf[valid] * w_old + tsdf_val[valid] * w_new
        ) / w_sum
        self.color[valid] = (
            self.color[valid] * w_old[..., None] + obs_color[valid] * w_new
        ) / w_sum[..., None]
        self.weight[valid] = w_sum

    def extract_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract mesh using marching cubes.

        Returns:
            vertices: (V, 3) vertex positions
            faces: (F, 3) triangle indices
            colors: (V, 3) vertex colors [0, 1]
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise RuntimeError(
                "scikit-image is required for mesh extraction. "
                "Install with: pip install scikit-image"
            )

        # Only extract where we have observations
        valid_mask = self.weight > 0
        if not valid_mask.any():
            raise RuntimeError("No valid TSDF observations for mesh extraction")

        # Marching cubes at zero-crossing
        vertices, faces, normals, _ = marching_cubes(
            self.tsdf,
            level=0.0,
            spacing=(self.voxel_size, self.voxel_size, self.voxel_size),
        )

        # Offset to world coordinates
        vertices += self.origin

        # Interpolate vertex colors from TSDF color volume
        voxel_coords = (
            (vertices - self.origin) / self.voxel_size
        ).astype(int)
        voxel_coords = np.clip(
            voxel_coords,
            0,
            np.array(self.dims) - 1,
        )
        colors = self.color[
            voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
        ]

        logger.info(
            f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces"
        )
        return vertices, faces, colors


def extract_mesh(
    trainer,
    output_path: Path,
    voxel_size: float = 0.02,
    progress_callback=None,
) -> dict:
    """Render depth maps from training views, fuse into TSDF, extract mesh.

    Args:
        trainer: GaussianTrainer with trained model and loaded cameras
        output_path: Path to save the output mesh (GLB format)
        voxel_size: TSDF voxel size in meters (default 2cm)
        progress_callback: Optional callback(fraction) for progress updates

    Returns:
        dict with vertex_count and face_count
    """
    try:
        import trimesh
    except ImportError:
        raise RuntimeError(
            "trimesh is required for mesh export. "
            "Install with: pip install trimesh"
        )

    device = trainer.device
    cameras = trainer.cameras_data

    # Compute scene bounds from Gaussian means
    with torch.no_grad():
        means = trainer.params["means"].detach().cpu().numpy()
        # Use percentile to exclude outliers
        bounds_min = np.percentile(means, 2, axis=0) - 0.5
        bounds_max = np.percentile(means, 98, axis=0) + 0.5

    volume = TSDFVolume(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        voxel_size=voxel_size,
    )

    # Render depth maps from all training views and integrate
    n_views = len(cameras)
    logger.info(f"Rendering {n_views} depth maps for TSDF fusion...")

    for i, cam in enumerate(cameras):
        W, H = cam["width"], cam["height"]
        viewmat = torch.tensor(cam["viewmat"], device=device).unsqueeze(0)
        K = torch.tensor(cam["K"], device=device).unsqueeze(0)

        with torch.no_grad():
            result = trainer.rasterizer.rasterize(
                trainer.params, viewmat, K, W, H, trainer.config.sh_degree
            )
            depth_np = result.depth.squeeze(-1).cpu().numpy()
            color_np = result.image.clamp(0, 1).cpu().numpy()

        volume.integrate(depth_np, color_np, cam["K"], cam["viewmat"])

        if progress_callback and i % 10 == 0:
            progress_callback((i + 1) / n_views * 0.8)

    # Extract mesh
    if progress_callback:
        progress_callback(0.85)

    vertices, faces, colors = volume.extract_mesh()

    # Create trimesh and export as GLB
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(colors * 255).astype(np.uint8),
    )

    if progress_callback:
        progress_callback(0.95)

    mesh.export(str(output_path))
    logger.info(f"Mesh exported to {output_path}")

    if progress_callback:
        progress_callback(1.0)

    return {
        "vertex_count": len(vertices),
        "face_count": len(faces),
    }
