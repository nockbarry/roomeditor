"""In-memory scene manager with delta-based undo/redo.

Holds gaussian data in numpy structured arrays. All edits mutate arrays in-place (<1ms).
PLY/SPZ only written on explicit save(). Delta undo records store per-edit diffs (~96KB per
1000-gaussian transform vs 1.6GB full file copy).
"""

import logging
import shutil
import tempfile
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

MAX_UNDO = 50
MAX_CACHED_SCENES = 3

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DeltaRecord:
    """Stores a reversible edit: indices + old/new values for each modified attribute."""
    label: str
    timestamp: float
    indices: np.ndarray  # uint32 array of affected gaussian indices
    old_values: dict[str, np.ndarray] = field(default_factory=dict)
    new_values: dict[str, np.ndarray] = field(default_factory=dict)
    # For delete/undelete: which indices changed in the deleted mask
    deleted_before: np.ndarray | None = None  # bool values at indices before
    deleted_after: np.ndarray | None = None   # bool values at indices after

    @property
    def size_bytes(self) -> int:
        """Estimate memory usage."""
        total = self.indices.nbytes
        for v in self.old_values.values():
            total += v.nbytes
        for v in self.new_values.values():
            total += v.nbytes
        if self.deleted_before is not None:
            total += self.deleted_before.nbytes
        if self.deleted_after is not None:
            total += self.deleted_after.nbytes
        return total


@dataclass
class TransformOp:
    """Transform gaussians: move/rotate/scale."""
    type: Literal["transform"] = "transform"
    indices: list[int] | np.ndarray = field(default_factory=list)
    translation: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None  # euler degrees
    scale: tuple[float, float, float] | None = None


@dataclass
class DeleteOp:
    """Soft-delete gaussians."""
    type: Literal["delete"] = "delete"
    indices: list[int] | np.ndarray = field(default_factory=list)


@dataclass
class UndeleteOp:
    """Restore soft-deleted gaussians."""
    type: Literal["undelete"] = "undelete"
    indices: list[int] | np.ndarray = field(default_factory=list)


@dataclass
class LightingOp:
    """Adjust SH coefficients for brightness/color."""
    type: Literal["lighting"] = "lighting"
    indices: list[int] | np.ndarray = field(default_factory=list)
    brightness: float = 1.0
    color_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sh_scale: float = 1.0


@dataclass
class PropertyEditOp:
    """Write arbitrary attribute values."""
    type: Literal["property"] = "property"
    indices: list[int] | np.ndarray = field(default_factory=list)
    attr: str = ""
    values: np.ndarray | None = None


EditOp = TransformOp | DeleteOp | UndeleteOp | LightingOp | PropertyEditOp


# ---------------------------------------------------------------------------
# SceneManager
# ---------------------------------------------------------------------------

class SceneManager:
    """Holds a project's gaussian scene in memory for fast editing."""

    def __init__(self, project_id: str, project_dir: Path):
        self.project_id = project_id
        self.project_dir = project_dir
        self.data: np.ndarray | None = None  # structured array from PLY
        self.deleted: np.ndarray | None = None  # bool mask (True = soft-deleted)
        self.n_gaussians: int = 0
        self.undo_stack: list[DeltaRecord] = []
        self.redo_stack: list[DeltaRecord] = []
        self.dirty: bool = False
        self._lock = threading.Lock()

    def load(self) -> dict:
        """Load scene.ply into memory. Returns scene info dict."""
        from plyfile import PlyData

        ply_path = self.project_dir / "scene.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"No scene.ply in {self.project_dir}")

        t0 = time.time()
        plydata = PlyData.read(str(ply_path))
        self.data = plydata["vertex"].data.copy()  # own the array (not mmap)
        self.n_gaussians = len(self.data)
        self.deleted = np.zeros(self.n_gaussians, dtype=bool)

        # Mark already-deleted gaussians (opacity = -100)
        if "opacity" in self.data.dtype.names:
            opacities = np.array(self.data["opacity"])
            self.deleted = opacities <= -50.0

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False

        # Compute bounds
        x = np.array(self.data["x"])
        y = np.array(self.data["y"])
        z = np.array(self.data["z"])
        bbox_min = [float(x.min()), float(y.min()), float(z.min())]
        bbox_max = [float(x.max()), float(y.max()), float(z.max())]

        duration = time.time() - t0
        logger.info(f"SceneManager loaded {self.n_gaussians:,} gaussians in {duration:.1f}s")

        return {
            "n_gaussians": self.n_gaussians,
            "n_deleted": int(self.deleted.sum()),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "load_time_sec": round(duration, 2),
        }

    def save(self) -> dict:
        """Write current state to PLY + regenerate SPZ."""
        if self.data is None:
            raise RuntimeError("No scene loaded")

        t0 = time.time()
        ply_path = self.project_dir / "scene.ply"

        # Apply soft deletes: set opacity = -100 for deleted gaussians
        data_copy = self.data.copy()
        if "opacity" in data_copy.dtype.names and self.deleted is not None:
            opacities = np.array(data_copy["opacity"])
            opacities[self.deleted] = -100.0
            data_copy["opacity"] = opacities

        from plyfile import PlyData, PlyElement
        element = PlyElement.describe(data_copy, "vertex")
        plydata = PlyData([element], text=False)

        # Write via temp file + rename
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=str(ply_path.parent))
        try:
            import os
            os.close(tmp_fd)
            plydata.write(tmp_path)
            shutil.move(tmp_path, str(ply_path))
        except Exception:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # Regenerate SPZ + positions sidecar
        from utils.spz_convert import generate_spz_bundle
        generate_spz_bundle(self.project_dir)

        self.dirty = False
        duration = time.time() - t0
        logger.info(f"SceneManager saved {self.n_gaussians:,} gaussians in {duration:.1f}s")

        return {
            "n_gaussians": self.n_gaussians,
            "save_time_sec": round(duration, 2),
        }

    def get_positions(self) -> np.ndarray:
        """Return Nx3 float32 positions array."""
        if self.data is None:
            raise RuntimeError("No scene loaded")
        return np.column_stack([
            np.array(self.data["x"]),
            np.array(self.data["y"]),
            np.array(self.data["z"]),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Edit operations
    # ------------------------------------------------------------------

    def apply_edit(self, op: EditOp) -> dict:
        """Apply an edit operation. Returns summary with undo/redo counts."""
        if self.data is None:
            raise RuntimeError("No scene loaded")

        with self._lock:
            if isinstance(op, TransformOp):
                delta = self._apply_transform(op)
            elif isinstance(op, DeleteOp):
                delta = self._apply_delete(op)
            elif isinstance(op, UndeleteOp):
                delta = self._apply_undelete(op)
            elif isinstance(op, LightingOp):
                delta = self._apply_lighting(op)
            elif isinstance(op, PropertyEditOp):
                delta = self._apply_property(op)
            else:
                raise ValueError(f"Unknown edit op type: {type(op)}")

            # Push to undo stack, clear redo
            self.undo_stack.append(delta)
            self.redo_stack.clear()
            if len(self.undo_stack) > MAX_UNDO:
                self.undo_stack.pop(0)
            self.dirty = True

        return {
            "label": delta.label,
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
            "n_affected": len(delta.indices),
        }

    def undo(self) -> dict | None:
        """Undo the last edit. Returns summary or None if nothing to undo."""
        if self.data is None or not self.undo_stack:
            return None

        with self._lock:
            delta = self.undo_stack.pop()
            self._reverse_delta(delta)
            self.redo_stack.append(delta)
            self.dirty = True

        return {
            "label": delta.label,
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
        }

    def redo(self) -> dict | None:
        """Redo the last undone edit. Returns summary or None if nothing to redo."""
        if self.data is None or not self.redo_stack:
            return None

        with self._lock:
            delta = self.redo_stack.pop()
            self._apply_delta_forward(delta)
            self.undo_stack.append(delta)
            self.dirty = True

        return {
            "label": delta.label,
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
        }

    def get_history(self) -> dict:
        """Return undo/redo stack labels."""
        return {
            "undo": [{"label": d.label, "n_affected": len(d.indices)} for d in reversed(self.undo_stack)],
            "redo": [{"label": d.label, "n_affected": len(d.indices)} for d in reversed(self.redo_stack)],
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
            "dirty": self.dirty,
        }

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def query_box(self, min_pt: tuple[float, float, float],
                  max_pt: tuple[float, float, float],
                  exclude_deleted: bool = True) -> np.ndarray:
        """Return indices of gaussians within an AABB."""
        if self.data is None:
            return np.array([], dtype=np.uint32)

        x = np.array(self.data["x"])
        y = np.array(self.data["y"])
        z = np.array(self.data["z"])

        mask = (
            (x >= min_pt[0]) & (x <= max_pt[0]) &
            (y >= min_pt[1]) & (y <= max_pt[1]) &
            (z >= min_pt[2]) & (z <= max_pt[2])
        )
        if exclude_deleted and self.deleted is not None:
            mask &= ~self.deleted

        return np.where(mask)[0].astype(np.uint32)

    def query_sphere(self, center: tuple[float, float, float],
                     radius: float,
                     exclude_deleted: bool = True) -> np.ndarray:
        """Return indices of gaussians within a sphere."""
        if self.data is None:
            return np.array([], dtype=np.uint32)

        x = np.array(self.data["x"]) - center[0]
        y = np.array(self.data["y"]) - center[1]
        z = np.array(self.data["z"]) - center[2]

        dist_sq = x * x + y * y + z * z
        mask = dist_sq <= radius * radius
        if exclude_deleted and self.deleted is not None:
            mask &= ~self.deleted

        return np.where(mask)[0].astype(np.uint32)

    def query_radius(self, point: tuple[float, float, float],
                     radius: float,
                     exclude_deleted: bool = True) -> np.ndarray:
        """Alias for query_sphere (brush tool)."""
        return self.query_sphere(point, radius, exclude_deleted)

    # ------------------------------------------------------------------
    # Internal: apply specific edit types
    # ------------------------------------------------------------------

    def _apply_transform(self, op: TransformOp) -> DeltaRecord:
        """Apply translation/rotation/scale to gaussians."""
        ids = np.asarray(op.indices, dtype=np.uint32)
        label_parts = []

        old_values: dict[str, np.ndarray] = {}
        new_values: dict[str, np.ndarray] = {}

        x = np.array(self.data["x"], dtype=np.float64)
        y = np.array(self.data["y"], dtype=np.float64)
        z = np.array(self.data["z"], dtype=np.float64)

        # Save old positions
        old_values["x"] = x[ids].astype(np.float32).copy()
        old_values["y"] = y[ids].astype(np.float32).copy()
        old_values["z"] = z[ids].astype(np.float32).copy()

        # Compute centroid for rotation/scale pivot
        cx, cy, cz = x[ids].mean(), y[ids].mean(), z[ids].mean()

        if op.rotation:
            rx, ry, rz = op.rotation
            if rx != 0 or ry != 0 or rz != 0:
                label_parts.append("rotate")
                from scipy.spatial.transform import Rotation as R
                rot = R.from_euler("xyz", [rx, ry, rz], degrees=True)
                rot_matrix = rot.as_matrix()

                positions = np.column_stack([x[ids] - cx, y[ids] - cy, z[ids] - cz])
                rotated = positions @ rot_matrix.T
                x[ids] = rotated[:, 0] + cx
                y[ids] = rotated[:, 1] + cy
                z[ids] = rotated[:, 2] + cz

                # Compose with existing per-gaussian quaternions
                quat_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
                if all(n in self.data.dtype.names for n in quat_names):
                    qw = np.array(self.data["rot_0"], dtype=np.float64)
                    qx = np.array(self.data["rot_1"], dtype=np.float64)
                    qy = np.array(self.data["rot_2"], dtype=np.float64)
                    qz = np.array(self.data["rot_3"], dtype=np.float64)

                    old_values["rot_0"] = qw[ids].astype(np.float32).copy()
                    old_values["rot_1"] = qx[ids].astype(np.float32).copy()
                    old_values["rot_2"] = qy[ids].astype(np.float32).copy()
                    old_values["rot_3"] = qz[ids].astype(np.float32).copy()

                    existing_quats = np.column_stack([qx[ids], qy[ids], qz[ids], qw[ids]])
                    existing_rots = R.from_quat(existing_quats)
                    new_rots = rot * existing_rots
                    new_quats = new_rots.as_quat()  # (x,y,z,w)

                    qw[ids] = new_quats[:, 3]
                    qx[ids] = new_quats[:, 0]
                    qy[ids] = new_quats[:, 1]
                    qz[ids] = new_quats[:, 2]

                    self.data["rot_0"] = qw.astype(np.float32)
                    self.data["rot_1"] = qx.astype(np.float32)
                    self.data["rot_2"] = qy.astype(np.float32)
                    self.data["rot_3"] = qz.astype(np.float32)

                    new_values["rot_0"] = qw[ids].astype(np.float32).copy()
                    new_values["rot_1"] = qx[ids].astype(np.float32).copy()
                    new_values["rot_2"] = qy[ids].astype(np.float32).copy()
                    new_values["rot_3"] = qz[ids].astype(np.float32).copy()

        if op.scale:
            sx, sy, sz = op.scale
            if sx != 1.0 or sy != 1.0 or sz != 1.0:
                label_parts.append("scale")
                # Recompute centroid after potential rotation
                cx, cy, cz = x[ids].mean(), y[ids].mean(), z[ids].mean()
                x[ids] = (x[ids] - cx) * sx + cx
                y[ids] = (y[ids] - cy) * sy + cy
                z[ids] = (z[ids] - cz) * sz + cz

                # Scale gaussian log-space scales
                for attr, sf in zip(["scale_0", "scale_1", "scale_2"], [sx, sy, sz]):
                    if attr in self.data.dtype.names and sf > 0:
                        vals = np.array(self.data[attr])
                        old_values[attr] = vals[ids].copy()
                        vals[ids] += np.log(sf).astype(np.float32)
                        self.data[attr] = vals
                        new_values[attr] = vals[ids].copy()

        if op.translation:
            tx, ty, tz = op.translation
            if tx != 0 or ty != 0 or tz != 0:
                label_parts.append("translate")
                x[ids] += tx
                y[ids] += ty
                z[ids] += tz

        # Write back positions
        self.data["x"] = x.astype(np.float32)
        self.data["y"] = y.astype(np.float32)
        self.data["z"] = z.astype(np.float32)

        new_values["x"] = x[ids].astype(np.float32).copy()
        new_values["y"] = y[ids].astype(np.float32).copy()
        new_values["z"] = z[ids].astype(np.float32).copy()

        label = f"Transform ({'+'.join(label_parts or ['noop'])}) {len(ids)} gaussians"
        return DeltaRecord(
            label=label,
            timestamp=time.time(),
            indices=ids,
            old_values=old_values,
            new_values=new_values,
        )

    def _apply_delete(self, op: DeleteOp) -> DeltaRecord:
        """Soft-delete gaussians by setting deleted mask."""
        ids = np.asarray(op.indices, dtype=np.uint32)
        deleted_before = self.deleted[ids].copy()
        self.deleted[ids] = True
        deleted_after = self.deleted[ids].copy()

        return DeltaRecord(
            label=f"Delete {len(ids)} gaussians",
            timestamp=time.time(),
            indices=ids,
            deleted_before=deleted_before,
            deleted_after=deleted_after,
        )

    def _apply_undelete(self, op: UndeleteOp) -> DeltaRecord:
        """Restore soft-deleted gaussians."""
        ids = np.asarray(op.indices, dtype=np.uint32)
        deleted_before = self.deleted[ids].copy()
        self.deleted[ids] = False
        deleted_after = self.deleted[ids].copy()

        return DeltaRecord(
            label=f"Undelete {len(ids)} gaussians",
            timestamp=time.time(),
            indices=ids,
            deleted_before=deleted_before,
            deleted_after=deleted_after,
        )

    def _apply_lighting(self, op: LightingOp) -> DeltaRecord:
        """Adjust SH DC coefficients for brightness/color."""
        ids = np.asarray(op.indices, dtype=np.uint32)
        old_values: dict[str, np.ndarray] = {}
        new_values: dict[str, np.ndarray] = {}

        # Modify DC SH coefficients
        for ch, (attr, tint) in enumerate(zip(
            ["f_dc_0", "f_dc_1", "f_dc_2"],
            op.color_tint,
        )):
            if attr in self.data.dtype.names:
                vals = np.array(self.data[attr], dtype=np.float64)
                old_values[attr] = vals[ids].astype(np.float32).copy()
                vals[ids] *= op.brightness * tint
                self.data[attr] = vals.astype(np.float32)
                new_values[attr] = vals[ids].astype(np.float32).copy()

        # Scale higher-order SH bands
        if op.sh_scale != 1.0:
            for i in range(45):
                attr = f"f_rest_{i}"
                if attr in self.data.dtype.names:
                    vals = np.array(self.data[attr], dtype=np.float64)
                    old_values[attr] = vals[ids].astype(np.float32).copy()
                    vals[ids] *= op.sh_scale
                    self.data[attr] = vals.astype(np.float32)
                    new_values[attr] = vals[ids].astype(np.float32).copy()

        return DeltaRecord(
            label=f"Lighting adjust {len(ids)} gaussians",
            timestamp=time.time(),
            indices=ids,
            old_values=old_values,
            new_values=new_values,
        )

    def _apply_property(self, op: PropertyEditOp) -> DeltaRecord:
        """Write arbitrary attribute values."""
        ids = np.asarray(op.indices, dtype=np.uint32)
        if op.attr not in self.data.dtype.names:
            raise ValueError(f"Unknown attribute: {op.attr}")

        vals = np.array(self.data[op.attr])
        old_values = {op.attr: vals[ids].copy()}
        vals[ids] = op.values
        self.data[op.attr] = vals
        new_values = {op.attr: vals[ids].copy()}

        return DeltaRecord(
            label=f"Set {op.attr} on {len(ids)} gaussians",
            timestamp=time.time(),
            indices=ids,
            old_values=old_values,
            new_values=new_values,
        )

    # ------------------------------------------------------------------
    # Internal: reverse / reapply deltas
    # ------------------------------------------------------------------

    def _reverse_delta(self, delta: DeltaRecord):
        """Undo a delta: restore old values."""
        ids = delta.indices
        for attr, old_vals in delta.old_values.items():
            vals = np.array(self.data[attr])
            vals[ids] = old_vals
            self.data[attr] = vals

        if delta.deleted_before is not None:
            self.deleted[ids] = delta.deleted_before

    def _apply_delta_forward(self, delta: DeltaRecord):
        """Redo a delta: apply new values."""
        ids = delta.indices
        for attr, new_vals in delta.new_values.items():
            vals = np.array(self.data[attr])
            vals[ids] = new_vals
            self.data[attr] = vals

        if delta.deleted_after is not None:
            self.deleted[ids] = delta.deleted_after


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

_scene_cache: OrderedDict[str, SceneManager] = OrderedDict()
_cache_lock = threading.Lock()


def get_scene(project_id: str, project_dir: Path, auto_load: bool = True) -> SceneManager:
    """Get or create a SceneManager for a project. Thread-safe with LRU eviction."""
    with _cache_lock:
        if project_id in _scene_cache:
            _scene_cache.move_to_end(project_id)
            return _scene_cache[project_id]

    scene = SceneManager(project_id, project_dir)
    if auto_load:
        scene.load()

    with _cache_lock:
        # Evict oldest if at capacity
        while len(_scene_cache) >= MAX_CACHED_SCENES:
            evicted_id, evicted_scene = _scene_cache.popitem(last=False)
            if evicted_scene.dirty:
                logger.info(f"Auto-saving evicted scene: {evicted_id}")
                try:
                    evicted_scene.save()
                except Exception as e:
                    logger.error(f"Failed to auto-save evicted scene {evicted_id}: {e}")

        _scene_cache[project_id] = scene

    return scene


def evict_scene(project_id: str, save: bool = True):
    """Remove a scene from cache, optionally saving first."""
    with _cache_lock:
        scene = _scene_cache.pop(project_id, None)
    if scene and save and scene.dirty:
        scene.save()


def get_cached_scene(project_id: str) -> SceneManager | None:
    """Get a cached scene without loading. Returns None if not in cache."""
    with _cache_lock:
        if project_id in _scene_cache:
            _scene_cache.move_to_end(project_id)
            return _scene_cache[project_id]
    return None
