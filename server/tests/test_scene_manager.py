"""Tests for pipeline.scene_manager — in-memory editing, undo/redo, spatial queries."""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: create a minimal PLY in a temp dir
# ---------------------------------------------------------------------------

def _make_ply(tmp_dir: Path, n: int = 100, *, with_quats: bool = True, with_sh: bool = True):
    """Write a synthetic scene.ply with n gaussians on a grid."""
    from plyfile import PlyData, PlyElement

    # Positions: grid along x-axis
    x = np.linspace(-5, 5, n).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    opacity = np.ones(n, dtype=np.float32)

    dtype_fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ]

    if with_quats:
        dtype_fields += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    if with_sh:
        dtype_fields += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]

    data = np.zeros(n, dtype=dtype_fields)
    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["opacity"] = opacity
    data["scale_0"] = np.full(n, -3.0, dtype=np.float32)
    data["scale_1"] = np.full(n, -3.0, dtype=np.float32)
    data["scale_2"] = np.full(n, -3.0, dtype=np.float32)

    if with_quats:
        data["rot_0"] = np.ones(n, dtype=np.float32)  # w=1
        data["rot_1"] = np.zeros(n, dtype=np.float32)
        data["rot_2"] = np.zeros(n, dtype=np.float32)
        data["rot_3"] = np.zeros(n, dtype=np.float32)

    if with_sh:
        data["f_dc_0"] = np.full(n, 0.5, dtype=np.float32)
        data["f_dc_1"] = np.full(n, 0.5, dtype=np.float32)
        data["f_dc_2"] = np.full(n, 0.5, dtype=np.float32)

    element = PlyElement.describe(data, "vertex")
    PlyData([element], text=False).write(str(tmp_dir / "scene.ply"))
    return data


@pytest.fixture
def scene_dir():
    """Create a temp directory with a synthetic PLY, clean up after test."""
    tmp = Path(tempfile.mkdtemp(prefix="test_scene_"))
    _make_ply(tmp, n=100)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def scene(scene_dir):
    """Return a loaded SceneManager."""
    from pipeline.scene_manager import SceneManager
    mgr = SceneManager("test-project", scene_dir)
    mgr.load()
    return mgr


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_basic(self, scene):
        assert scene.data is not None
        assert scene.n_gaussians == 100
        assert scene.deleted is not None
        assert scene.deleted.sum() == 0
        assert not scene.dirty

    def test_load_returns_info(self, scene_dir):
        from pipeline.scene_manager import SceneManager
        mgr = SceneManager("test", scene_dir)
        info = mgr.load()
        assert info["n_gaussians"] == 100
        assert info["n_deleted"] == 0
        assert len(info["bbox_min"]) == 3
        assert len(info["bbox_max"]) == 3
        assert info["bbox_min"][0] < info["bbox_max"][0]

    def test_load_detects_soft_deleted(self, scene_dir):
        """Gaussians with opacity <= -50 should be marked as deleted."""
        from plyfile import PlyData, PlyElement
        from pipeline.scene_manager import SceneManager

        # Modify the PLY to have some deleted gaussians
        plydata = PlyData.read(str(scene_dir / "scene.ply"))
        data = plydata["vertex"].data.copy()
        # Set first 10 gaussians as soft-deleted
        opacities = np.array(data["opacity"])
        opacities[:10] = -100.0
        data["opacity"] = opacities
        element = PlyElement.describe(data, "vertex")
        PlyData([element], text=False).write(str(scene_dir / "scene.ply"))

        mgr = SceneManager("test", scene_dir)
        info = mgr.load()
        assert info["n_deleted"] == 10
        assert mgr.deleted[:10].all()
        assert not mgr.deleted[10:].any()

    def test_load_missing_file(self):
        from pipeline.scene_manager import SceneManager
        tmp = Path(tempfile.mkdtemp())
        try:
            mgr = SceneManager("test", tmp)
            with pytest.raises(FileNotFoundError):
                mgr.load()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

class TestPositions:
    def test_get_positions(self, scene):
        pos = scene.get_positions()
        assert pos.shape == (100, 3)
        assert pos.dtype == np.float32
        # First point should be around -5, last around 5
        assert pos[0, 0] < -4.0
        assert pos[-1, 0] > 4.0

    def test_get_positions_not_loaded(self, scene_dir):
        from pipeline.scene_manager import SceneManager
        mgr = SceneManager("test", scene_dir)
        with pytest.raises(RuntimeError):
            mgr.get_positions()


# ---------------------------------------------------------------------------
# Delete / Undelete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_soft_delete(self, scene):
        from pipeline.scene_manager import DeleteOp
        result = scene.apply_edit(DeleteOp(indices=[0, 1, 2]))
        assert result["n_affected"] == 3
        assert result["undo_count"] == 1
        assert result["redo_count"] == 0
        assert scene.deleted[0] and scene.deleted[1] and scene.deleted[2]
        assert not scene.deleted[3]
        assert scene.dirty

    def test_undelete(self, scene):
        from pipeline.scene_manager import DeleteOp, UndeleteOp
        scene.apply_edit(DeleteOp(indices=[5, 6]))
        assert scene.deleted[5]
        scene.apply_edit(UndeleteOp(indices=[5, 6]))
        assert not scene.deleted[5]
        assert not scene.deleted[6]

    def test_delete_undo(self, scene):
        from pipeline.scene_manager import DeleteOp
        scene.apply_edit(DeleteOp(indices=[10, 11, 12]))
        assert scene.deleted[10:13].all()
        scene.undo()
        assert not scene.deleted[10:13].any()

    def test_delete_redo(self, scene):
        from pipeline.scene_manager import DeleteOp
        scene.apply_edit(DeleteOp(indices=[20, 21]))
        scene.undo()
        assert not scene.deleted[20]
        scene.redo()
        assert scene.deleted[20] and scene.deleted[21]


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TestTransform:
    def test_translate(self, scene):
        from pipeline.scene_manager import TransformOp
        old_x = float(scene.data["x"][0])
        result = scene.apply_edit(TransformOp(
            indices=[0, 1, 2],
            translation=(10.0, 0.0, 0.0),
        ))
        assert result["n_affected"] == 3
        assert "translate" in result["label"]
        new_x = float(scene.data["x"][0])
        assert abs(new_x - old_x - 10.0) < 0.01

    def test_translate_undo(self, scene):
        from pipeline.scene_manager import TransformOp
        old_x = float(scene.data["x"][0])
        scene.apply_edit(TransformOp(
            indices=[0],
            translation=(5.0, 3.0, 1.0),
        ))
        assert abs(float(scene.data["x"][0]) - old_x - 5.0) < 0.01
        scene.undo()
        assert abs(float(scene.data["x"][0]) - old_x) < 0.001

    def test_scale(self, scene):
        from pipeline.scene_manager import TransformOp
        old_scale0 = float(scene.data["scale_0"][0])
        scene.apply_edit(TransformOp(
            indices=[0, 1, 2],
            scale=(2.0, 2.0, 2.0),
        ))
        new_scale0 = float(scene.data["scale_0"][0])
        assert abs(new_scale0 - old_scale0 - np.log(2.0)) < 0.01

    def test_rotation_with_quaternions(self, scene):
        from pipeline.scene_manager import TransformOp
        # 90 degree rotation around z
        scene.apply_edit(TransformOp(
            indices=list(range(10)),
            rotation=(0.0, 0.0, 90.0),
        ))
        # Quaternions should have changed
        qw = float(scene.data["rot_0"][0])
        qz = float(scene.data["rot_3"][0])
        # Original was identity (1,0,0,0). After 90 deg around z: cos(45)=~0.707, sin(45)=~0.707
        assert abs(qw - 0.707) < 0.01
        assert abs(qz - 0.707) < 0.01


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

class TestLighting:
    def test_brightness_adjust(self, scene):
        from pipeline.scene_manager import LightingOp
        old_dc0 = float(scene.data["f_dc_0"][0])
        scene.apply_edit(LightingOp(
            indices=[0, 1, 2],
            brightness=2.0,
            color_tint=(1.0, 1.0, 1.0),
        ))
        new_dc0 = float(scene.data["f_dc_0"][0])
        assert abs(new_dc0 - old_dc0 * 2.0) < 0.01

    def test_color_tint(self, scene):
        from pipeline.scene_manager import LightingOp
        old_dc0 = float(scene.data["f_dc_0"][0])
        old_dc1 = float(scene.data["f_dc_1"][0])
        scene.apply_edit(LightingOp(
            indices=[0],
            brightness=1.0,
            color_tint=(2.0, 0.5, 1.0),
        ))
        assert abs(float(scene.data["f_dc_0"][0]) - old_dc0 * 2.0) < 0.01
        assert abs(float(scene.data["f_dc_1"][0]) - old_dc1 * 0.5) < 0.01


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------

class TestProperty:
    def test_set_opacity(self, scene):
        from pipeline.scene_manager import PropertyEditOp
        scene.apply_edit(PropertyEditOp(
            indices=[0, 1],
            attr="opacity",
            values=np.array([0.5, 0.3], dtype=np.float32),
        ))
        assert abs(float(scene.data["opacity"][0]) - 0.5) < 0.01
        assert abs(float(scene.data["opacity"][1]) - 0.3) < 0.01

    def test_unknown_attr_raises(self, scene):
        from pipeline.scene_manager import PropertyEditOp
        with pytest.raises(ValueError, match="Unknown attribute"):
            scene.apply_edit(PropertyEditOp(
                indices=[0],
                attr="nonexistent",
                values=np.array([1.0]),
            ))


# ---------------------------------------------------------------------------
# Undo / Redo
# ---------------------------------------------------------------------------

class TestUndoRedo:
    def test_multiple_undos(self, scene):
        from pipeline.scene_manager import DeleteOp
        scene.apply_edit(DeleteOp(indices=[0]))
        scene.apply_edit(DeleteOp(indices=[1]))
        scene.apply_edit(DeleteOp(indices=[2]))

        assert scene.deleted[0] and scene.deleted[1] and scene.deleted[2]

        scene.undo()
        assert not scene.deleted[2]
        assert scene.deleted[0] and scene.deleted[1]

        scene.undo()
        assert not scene.deleted[1]
        assert scene.deleted[0]

        scene.undo()
        assert not scene.deleted[0]

    def test_undo_nothing(self, scene):
        result = scene.undo()
        assert result is None

    def test_redo_nothing(self, scene):
        result = scene.redo()
        assert result is None

    def test_edit_clears_redo(self, scene):
        from pipeline.scene_manager import DeleteOp
        scene.apply_edit(DeleteOp(indices=[0]))
        scene.undo()
        assert len(scene.redo_stack) == 1
        scene.apply_edit(DeleteOp(indices=[1]))
        assert len(scene.redo_stack) == 0

    def test_undo_redo_cycle(self, scene):
        """Full cycle: edit → undo → redo should restore same state."""
        from pipeline.scene_manager import TransformOp
        old_x = float(scene.data["x"][5])
        scene.apply_edit(TransformOp(indices=[5], translation=(7.0, 0.0, 0.0)))
        moved_x = float(scene.data["x"][5])
        assert abs(moved_x - old_x - 7.0) < 0.01

        scene.undo()
        assert abs(float(scene.data["x"][5]) - old_x) < 0.001

        scene.redo()
        assert abs(float(scene.data["x"][5]) - moved_x) < 0.001

    def test_max_undo(self, scene):
        from pipeline.scene_manager import DeleteOp, MAX_UNDO
        for i in range(MAX_UNDO + 10):
            scene.apply_edit(DeleteOp(indices=[i % scene.n_gaussians]))
        assert len(scene.undo_stack) == MAX_UNDO

    def test_history(self, scene):
        from pipeline.scene_manager import DeleteOp, TransformOp
        scene.apply_edit(DeleteOp(indices=[0]))
        scene.apply_edit(TransformOp(indices=[1], translation=(1.0, 0.0, 0.0)))
        scene.undo()

        hist = scene.get_history()
        assert hist["undo_count"] == 1
        assert hist["redo_count"] == 1
        assert hist["dirty"]
        assert len(hist["undo"]) == 1
        assert len(hist["redo"]) == 1


# ---------------------------------------------------------------------------
# Spatial queries
# ---------------------------------------------------------------------------

class TestSpatialQueries:
    def test_query_box(self, scene):
        # Points are on x-axis from -5 to 5
        indices = scene.query_box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        assert len(indices) > 0
        # All returned points should be in the box
        for idx in indices:
            assert -1.0 <= float(scene.data["x"][idx]) <= 1.0

    def test_query_box_excludes_deleted(self, scene):
        from pipeline.scene_manager import DeleteOp
        indices_before = scene.query_box((-6.0, -1.0, -1.0), (6.0, 1.0, 1.0))
        # Delete first 10
        scene.apply_edit(DeleteOp(indices=list(range(10))))
        indices_after = scene.query_box((-6.0, -1.0, -1.0), (6.0, 1.0, 1.0))
        assert len(indices_after) < len(indices_before)

    def test_query_box_include_deleted(self, scene):
        from pipeline.scene_manager import DeleteOp
        scene.apply_edit(DeleteOp(indices=list(range(10))))
        indices = scene.query_box((-6.0, -1.0, -1.0), (6.0, 1.0, 1.0), exclude_deleted=False)
        assert len(indices) == 100  # all on axis

    def test_query_sphere(self, scene):
        indices = scene.query_sphere((0.0, 0.0, 0.0), 1.0)
        assert len(indices) > 0
        for idx in indices:
            x = float(scene.data["x"][idx])
            assert x * x <= 1.0 + 0.01  # within radius

    def test_query_sphere_empty(self, scene):
        # No points at y=100
        indices = scene.query_sphere((0.0, 100.0, 0.0), 0.1)
        assert len(indices) == 0

    def test_query_radius_alias(self, scene):
        a = scene.query_sphere((0.0, 0.0, 0.0), 2.0)
        b = scene.query_radius((0.0, 0.0, 0.0), 2.0)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_get_scene(self, scene_dir):
        from pipeline.scene_manager import get_scene, _scene_cache, _cache_lock

        # Clean cache first
        with _cache_lock:
            _scene_cache.clear()

        mgr = get_scene("cache-test", scene_dir)
        assert mgr.n_gaussians == 100
        assert "cache-test" in _scene_cache

        # Second call should return same instance
        mgr2 = get_scene("cache-test", scene_dir)
        assert mgr2 is mgr

        # Cleanup
        with _cache_lock:
            _scene_cache.clear()

    def test_get_cached_scene(self, scene_dir):
        from pipeline.scene_manager import get_scene, get_cached_scene, _scene_cache, _cache_lock

        with _cache_lock:
            _scene_cache.clear()

        assert get_cached_scene("not-loaded") is None

        get_scene("cached-test", scene_dir)
        mgr = get_cached_scene("cached-test")
        assert mgr is not None
        assert mgr.n_gaussians == 100

        with _cache_lock:
            _scene_cache.clear()

    def test_evict_scene(self, scene_dir):
        from pipeline.scene_manager import get_scene, evict_scene, get_cached_scene, _scene_cache, _cache_lock

        with _cache_lock:
            _scene_cache.clear()

        get_scene("evict-test", scene_dir)
        assert get_cached_scene("evict-test") is not None

        evict_scene("evict-test", save=False)
        assert get_cached_scene("evict-test") is None
