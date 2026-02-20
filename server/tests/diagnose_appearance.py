"""Diagnose appearance model and render corrected previews."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path("/home/nock/projects/roomeditor/server/data/projects/d7a93e8d-a6fa-46eb-b1da-199aba06559a")
COLMAP_DIR = PROJECT_DIR / "colmap" / "sparse" / "0"
FRAMES_DIR = PROJECT_DIR / "frames"
OUTPUT_DIR = PROJECT_DIR / "previews"
OUTPUT_DIR.mkdir(exist_ok=True)

from pipeline.train_gaussians import GaussianTrainer, TrainerConfig
from pipeline.rasterizer import _ensure_batched, rasterization_2dgs
from PIL import Image
import io

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

ckpt_path = PROJECT_DIR / "checkpoint_10000.pt"
print(f"Loading checkpoint: {ckpt_path}")
trainer.load_checkpoint(ckpt_path)
print(f"Resumed at step {trainer.current_step}, {len(trainer.params['means'])} Gaussians")

# ============================================================
# 1. Analyze appearance transforms
# ============================================================
print("\n=== Appearance Model Analysis ===")
identity = torch.eye(3).flatten().cuda()
with torch.no_grad():
    matrices = []
    biases = []
    for i in range(len(trainer.cameras_data)):
        embed = trainer.appearance_embeds(torch.tensor(i, device="cuda"))
        affine = trainer.appearance_mlp(embed)
        color_matrix = affine[:9].reshape(3, 3)
        color_bias = affine[9:]
        matrices.append(color_matrix)
        biases.append(color_bias)

    matrices = torch.stack(matrices)  # (N, 3, 3)
    biases = torch.stack(biases)      # (N, 3)

    # How far from identity?
    identity_mat = torch.eye(3, device="cuda")
    deviations = (matrices - identity_mat.unsqueeze(0)).abs().mean(dim=(1, 2))
    print(f"Mean matrix deviation from identity: {deviations.mean():.4f}")
    print(f"Max matrix deviation from identity:  {deviations.max():.4f}")
    print(f"Mean bias magnitude: {biases.abs().mean():.4f}")
    print(f"Max bias magnitude:  {biases.abs().max():.4f}")

    # Show a few matrices
    for i in [0, 25, 50, 75, 99]:
        print(f"\nView {i}:")
        print(f"  Matrix diag: {matrices[i].diag().cpu().numpy()}")
        print(f"  Bias: {biases[i].cpu().numpy()}")

    # Compute average transform
    mean_matrix = matrices.mean(dim=0)
    mean_bias = biases.mean(dim=0)
    print(f"\nMean matrix diag: {mean_matrix.diag().cpu().numpy()}")
    print(f"Mean bias: {mean_bias.cpu().numpy()}")

# ============================================================
# 2. Render with and without appearance correction
# ============================================================
print("\n=== Rendering comparisons ===")
view_indices = [0, 25, 50, 75, 99]

for idx in view_indices:
    cam = trainer.cameras_data[idx]
    W, H = cam["width"], cam["height"]
    device = trainer.params["means"].device
    viewmat, K = _ensure_batched(cam["viewmat"], cam["K"], device)

    with torch.no_grad():
        colors, *_ = rasterization_2dgs(
            means=trainer.params["means"],
            quats=trainer.params["quats"],
            scales=torch.exp(trainer.params["scales"]),
            opacities=torch.sigmoid(trainer.params["opacities"].squeeze(-1)),
            colors=torch.cat([trainer.params["sh0"], trainer.params["shN"]], dim=1),
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=config.sh_degree,
            packed=False,
            render_mode="RGB",
            distloss=False,
        )
        raw_img = colors[0, ..., :3].clamp(0, 1)

        # Apply per-view appearance correction
        embed = trainer.appearance_embeds(torch.tensor(idx, device=device))
        affine = trainer.appearance_mlp(embed)
        color_matrix = affine[:9].reshape(3, 3)
        color_bias = affine[9:]
        corrected_img = (raw_img @ color_matrix.T) + color_bias
        corrected_img = corrected_img.clamp(0, 1)

        # Apply mean appearance correction
        mean_corrected = (raw_img @ mean_matrix.T) + mean_bias
        mean_corrected = mean_corrected.clamp(0, 1)

    # Save raw
    raw_np = (raw_img.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(raw_np).save(OUTPUT_DIR / f"raw_cam{idx:03d}.jpg", quality=85)

    # Save per-view corrected
    corr_np = (corrected_img.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(corr_np).save(OUTPUT_DIR / f"corrected_cam{idx:03d}.jpg", quality=85)

    # Save mean corrected
    mean_np = (mean_corrected.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mean_np).save(OUTPUT_DIR / f"mean_corrected_cam{idx:03d}.jpg", quality=85)

    print(f"View {idx}: raw, corrected, mean_corrected saved")

print(f"\nAll saved to {OUTPUT_DIR}")
trainer.cleanup()
