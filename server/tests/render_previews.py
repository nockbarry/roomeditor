"""Render preview images from the final trained model."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path("/home/nock/projects/roomeditor/server/data/projects/d7a93e8d-a6fa-46eb-b1da-199aba06559a")
COLMAP_DIR = PROJECT_DIR / "colmap" / "sparse" / "0"
FRAMES_DIR = PROJECT_DIR / "frames"
OUTPUT_PLY = PROJECT_DIR / "scene_v2_2dgs.ply"
OUTPUT_DIR = PROJECT_DIR / "previews"
OUTPUT_DIR.mkdir(exist_ok=True)

from pipeline.train_gaussians import GaussianTrainer, TrainerConfig

config = TrainerConfig(
    iterations=15_000,
    sh_degree=2,
    mode="2dgs",
    appearance_embeddings=True,
    tidi_pruning=True,
)

trainer = GaussianTrainer(config)
points3d, colors3d = trainer.load_data(COLMAP_DIR, FRAMES_DIR)
trainer.init_params(points3d, colors3d)
trainer.init_strategy()

# Load the final checkpoint
ckpt_path = PROJECT_DIR / "checkpoint_10000.pt"
if ckpt_path.exists():
    print(f"Loading checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)
    print(f"Resumed at step {trainer.current_step}, {len(trainer.params['means'])} Gaussians")

# Or load directly from the PLY if no checkpoint
# Render previews from several viewpoints
n_views = len(trainer.cameras_data)
# Pick evenly spaced views
view_indices = [0, n_views // 6, n_views // 3, n_views // 2, 2 * n_views // 3, 5 * n_views // 6, n_views - 1]

for i, idx in enumerate(view_indices):
    cam = trainer.cameras_data[idx]
    jpeg_bytes = trainer.rasterizer.rasterize_preview(
        trainer.params,
        cam["viewmat"],
        cam["K"],
        cam["width"],
        cam["height"],
        trainer.config.sh_degree,
    )
    out_path = OUTPUT_DIR / f"view_{i:02d}_cam{idx:03d}.jpg"
    with open(out_path, "wb") as f:
        f.write(jpeg_bytes)
    print(f"Saved {out_path.name} ({len(jpeg_bytes)} bytes)")

print(f"\nAll previews saved to {OUTPUT_DIR}")
trainer.cleanup()
