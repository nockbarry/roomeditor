"""Tests for pipeline.extract_mesh module."""

import numpy as np
import pytest


class TestTSDFVolume:
    def test_init(self):
        from pipeline.extract_mesh import TSDFVolume
        vol = TSDFVolume(
            bounds_min=np.array([-1, -1, -1], dtype=np.float32),
            bounds_max=np.array([1, 1, 1], dtype=np.float32),
            voxel_size=0.1,
        )
        assert vol.dims[0] == 20
        assert vol.tsdf.shape == tuple(vol.dims)
        assert vol.weight.shape == tuple(vol.dims)

    def test_auto_resize_large_volume(self):
        from pipeline.extract_mesh import TSDFVolume
        # This would be 10000^3 at 0.001 → should auto-resize
        vol = TSDFVolume(
            bounds_min=np.array([-5, -5, -5], dtype=np.float32),
            bounds_max=np.array([5, 5, 5], dtype=np.float32),
            voxel_size=0.001,
        )
        assert max(vol.dims) <= 512

    def test_integrate_basic(self):
        from pipeline.extract_mesh import TSDFVolume
        vol = TSDFVolume(
            bounds_min=np.array([-1, -1, -1], dtype=np.float32),
            bounds_max=np.array([1, 1, 1], dtype=np.float32),
            voxel_size=0.1,
        )
        # Synthetic depth map: flat plane at z=2
        H, W = 64, 64
        depth = np.full((H, W), 2.0, dtype=np.float32)
        color = np.ones((H, W, 3), dtype=np.float32) * 0.5
        K = np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]], dtype=np.float32)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[2, 3] = 0.0  # Camera at origin

        vol.integrate(depth, color, K, viewmat)
        # At least some voxels should have been observed
        assert (vol.weight > 0).sum() > 0

    def test_extract_mesh_from_sphere_depth(self):
        """Verify marching cubes runs end-to-end on synthetic data."""
        pytest.importorskip("skimage")
        from pipeline.extract_mesh import TSDFVolume

        vol = TSDFVolume(
            bounds_min=np.array([-1, -1, 0], dtype=np.float32),
            bounds_max=np.array([1, 1, 4], dtype=np.float32),
            voxel_size=0.1,
        )

        # Integrate a single flat depth from a frontal camera
        H, W = 128, 128
        depth = np.full((H, W), 2.0, dtype=np.float32)
        color = np.ones((H, W, 3), dtype=np.float32) * 0.7
        K = np.array([[200, 0, 64], [0, 200, 64], [0, 0, 1]], dtype=np.float32)
        viewmat = np.eye(4, dtype=np.float32)

        vol.integrate(depth, color, K, viewmat)

        vertices, faces, colors = vol.extract_mesh()
        assert len(vertices) > 0
        assert len(faces) > 0
        assert colors.shape[1] == 3
