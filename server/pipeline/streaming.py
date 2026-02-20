"""Streaming incremental reconstruction pipeline.

Enables real-time 3D reconstruction as frames arrive from a phone camera:
1. Phone streams frames via WebSocket
2. MASt3R processes pairs on arrival for instant pose + depth
3. Gaussians initialized from depth maps for immediate coarse 3D
4. Background refinement runs continuously, improving quality
5. Progressive PLY updates pushed to the viewer

This module manages the incremental state machine that bridges
frame-by-frame MASt3R processing with continuous Gaussian optimization.
"""

import asyncio
import logging
import math
import struct
import sys
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from time import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming reconstruction."""
    mast3r_image_size: int = 512
    min_frames_to_start: int = 3          # Minimum frames before starting Gaussian training
    refine_iterations_per_batch: int = 500  # Training iterations per new frame batch
    max_gaussians: int = 500_000           # Cap for memory management
    sh_degree: int = 2
    mode: str = "2dgs"
    export_every_n_frames: int = 2         # Export PLY every N new frames


@dataclass
class StreamingState:
    """Mutable state for an in-progress streaming reconstruction."""
    project_id: str
    project_dir: Path
    config: StreamingConfig

    # Frame accumulation
    frames: list[Path] = field(default_factory=list)
    frame_count: int = 0

    # MASt3R state (pose estimates so far)
    poses_w2c: list[np.ndarray] = field(default_factory=list)
    intrinsics: list[np.ndarray] = field(default_factory=list)
    points3d_accumulated: np.ndarray | None = None
    colors3d_accumulated: np.ndarray | None = None

    # Gaussian training state
    trainer: object | None = None  # GaussianTrainer instance
    training_step: int = 0
    is_training: bool = False

    # Status
    phase: str = "waiting"  # waiting, accumulating, initializing, refining, done
    last_export_frame: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class StreamingReconstructor:
    """Manages incremental reconstruction from a stream of frames.

    Usage:
        reconstructor = StreamingReconstructor(config, project_dir)
        # As frames arrive:
        await reconstructor.add_frame(frame_path)
        # Periodically:
        ply_bytes = reconstructor.get_current_ply()
    """

    def __init__(
        self,
        config: StreamingConfig,
        project_id: str,
        project_dir: Path,
        on_progress: Callable | None = None,
        on_preview: Callable | None = None,
        on_ply_update: Callable | None = None,
    ):
        self.state = StreamingState(
            project_id=project_id,
            project_dir=project_dir,
            config=config,
        )
        self.on_progress = on_progress
        self.on_preview = on_preview
        self.on_ply_update = on_ply_update
        self._mast3r_model = None
        self._refine_task: asyncio.Task | None = None

    async def add_frame(self, frame_path: Path) -> dict:
        """Process a new frame from the camera stream.

        Returns a status dict with current phase and stats.
        """
        state = self.state

        with state.lock:
            state.frames.append(frame_path)
            state.frame_count += 1
            n = state.frame_count

        logger.info(f"Stream: received frame {n} ({frame_path.name})")

        if n < state.config.min_frames_to_start:
            state.phase = "accumulating"
            return self._status(f"Collecting frames ({n}/{state.config.min_frames_to_start})")

        if n == state.config.min_frames_to_start:
            # First batch — run full MASt3R + initialize Gaussians
            state.phase = "initializing"
            await self._initialize_from_frames()
            return self._status("Initial reconstruction complete")

        # Incremental update — add new frame to existing reconstruction
        state.phase = "refining"
        await self._add_incremental_frame(frame_path)

        # Check if we should export
        if n - state.last_export_frame >= state.config.export_every_n_frames:
            await self._export_current()
            state.last_export_frame = n

        return self._status(f"Refining ({n} frames, step {state.training_step})")

    async def _initialize_from_frames(self):
        """Run MASt3R on initial batch and start Gaussian training."""
        import torch

        state = self.state
        frames_dir = state.project_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Run MASt3R on all frames so far
        logger.info(f"Stream: initializing from {len(state.frames)} frames")

        from pipeline.run_mast3r import run_mast3r

        colmap_dir = state.project_dir / "colmap"
        model_dir, meta = run_mast3r(
            images_dir=frames_dir,
            output_dir=colmap_dir,
            image_size=state.config.mast3r_image_size,
        )

        if self.on_progress:
            await self.on_progress(0.3, "MASt3R pose estimation complete")

        # Initialize Gaussian trainer
        from pipeline.train_gaussians import GaussianTrainer, TrainerConfig, TrainerCallbacks

        config = TrainerConfig(
            iterations=state.config.refine_iterations_per_batch,
            sh_degree=state.config.sh_degree,
            mode=state.config.mode,
            depth_reg_weight=0.1,
        )

        trainer = GaussianTrainer(config)
        points3d, colors3d = trainer.load_data(model_dir, frames_dir)
        trainer.init_params(points3d, colors3d)
        trainer.init_strategy()

        state.trainer = trainer
        state.points3d_accumulated = points3d
        state.colors3d_accumulated = colors3d

        if self.on_progress:
            await self.on_progress(0.4, "Gaussians initialized")

        # Run initial refinement
        await self._run_refinement(state.config.refine_iterations_per_batch)

        # Export initial PLY
        await self._export_current()
        state.last_export_frame = state.frame_count

        if self.on_progress:
            await self.on_progress(0.6, "Initial refinement complete")

    async def _add_incremental_frame(self, frame_path: Path):
        """Add a single new frame incrementally.

        Re-runs MASt3R with all frames to get updated poses, then
        continues Gaussian training with the new view added.
        """
        state = self.state

        if state.trainer is None:
            return

        # For now, re-run MASt3R with all frames to get globally consistent poses.
        # Future optimization: use incremental pose estimation.
        frames_dir = state.project_dir / "frames"
        colmap_dir = state.project_dir / "colmap"

        from pipeline.run_mast3r import run_mast3r

        model_dir, meta = run_mast3r(
            images_dir=frames_dir,
            output_dir=colmap_dir,
            image_size=state.config.mast3r_image_size,
        )

        # Reload data with updated poses
        trainer = state.trainer
        from pipeline.train_gaussians import load_colmap_data

        cameras_data, new_points, new_colors = load_colmap_data(model_dir, frames_dir)

        # Update trainer's camera data and images
        trainer.cameras_data = cameras_data

        # Reload images
        from pipeline.train_gaussians import _load_image
        trainer.gt_images = []
        for cam in cameras_data:
            trainer.gt_images.append(
                _load_image(cam["image_path"], cam["width"], cam["height"])
            )

        # Continue training with updated views
        await self._run_refinement(state.config.refine_iterations_per_batch)

    async def _run_refinement(self, n_iterations: int):
        """Run N iterations of Gaussian training."""
        state = self.state
        trainer = state.trainer

        if trainer is None:
            return

        loop = asyncio.get_event_loop()

        def _train_batch():
            import torch

            start_step = state.training_step
            end_step = start_step + n_iterations

            # Temporarily set trainer config iterations to allow training
            old_iter = trainer.config.iterations
            trainer.config.iterations = end_step
            trainer.current_step = start_step

            for step in range(start_step, end_step):
                losses = trainer.train_step(step)
                state.training_step = step + 1

                if step % 100 == 0:
                    n_gs = len(trainer.params["means"])
                    logger.info(
                        f"  Stream train step {step}: loss={losses['total']:.4f}, N={n_gs}"
                    )

            trainer.config.iterations = old_iter
            state.is_training = False

        state.is_training = True
        await loop.run_in_executor(None, _train_batch)

    async def _export_current(self):
        """Export current Gaussians as PLY and notify client."""
        state = self.state
        trainer = state.trainer

        if trainer is None:
            return

        ply_path = state.project_dir / "scene.ply"
        trainer.export(ply_path)

        n_gs = len(trainer.params["means"])
        logger.info(f"Stream: exported {n_gs} Gaussians to {ply_path}")

        if self.on_ply_update:
            await self.on_ply_update(
                state.frame_count,
                state.training_step,
                n_gs,
                str(ply_path),
            )

    def _status(self, message: str) -> dict:
        state = self.state
        n_gs = 0
        if state.trainer and hasattr(state.trainer, 'params') and 'means' in state.trainer.params:
            n_gs = len(state.trainer.params["means"])

        return {
            "phase": state.phase,
            "frame_count": state.frame_count,
            "training_step": state.training_step,
            "num_gaussians": n_gs,
            "message": message,
        }

    def cleanup(self):
        """Free GPU memory."""
        if self.state.trainer:
            self.state.trainer.cleanup()
            self.state.trainer = None


# Global registry of active streaming sessions
_active_sessions: dict[str, StreamingReconstructor] = {}


def get_session(project_id: str) -> StreamingReconstructor | None:
    """Get an active streaming session by project ID."""
    return _active_sessions.get(project_id)


def create_session(
    project_id: str,
    project_dir: Path,
    config: StreamingConfig | None = None,
    **kwargs,
) -> StreamingReconstructor:
    """Create a new streaming reconstruction session."""
    if project_id in _active_sessions:
        _active_sessions[project_id].cleanup()

    if config is None:
        config = StreamingConfig()

    session = StreamingReconstructor(
        config=config,
        project_id=project_id,
        project_dir=project_dir,
        **kwargs,
    )
    _active_sessions[project_id] = session
    return session


def remove_session(project_id: str):
    """Remove and clean up a streaming session."""
    session = _active_sessions.pop(project_id, None)
    if session:
        session.cleanup()
