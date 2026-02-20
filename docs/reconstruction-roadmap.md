# Room Editor: Reconstruction Approach Comparison & Roadmap

## Current System (v2 — Per-Scene Optimization)

**Pipeline:** Phone video → ffmpeg frames → COLMAP SfM → gsplat 2DGS training → PLY export
**Training:** 15k iterations, ~16 min on RTX 4070 Ti SUPER
**Input:** ~100 video frames
**Output:** 318k Gaussians, 46 MB PLY
**Quality:** Recognizable room structure, blurry details, needs tuning

### Limitations
- Slow (COLMAP + training = 20+ min total)
- Requires dense overlapping frames for COLMAP
- No learned priors — reconstructs from scratch every time
- Phone video quality issues (motion blur, rolling shutter, auto-exposure)
- No quantitative evaluation (eyeballing only)

---

## Option A: Improve Current Pipeline (Quick Wins)

**Effort:** Low | **Impact:** Moderate

Changes:
- Implement PSNR/SSIM/LPIPS evaluation with held-out test views
- Smarter frame subsampling (keyframe extraction instead of every frame)
- Increase to 30k iterations, SH degree 3
- Test on Mip-NeRF 360 benchmark to compare against published numbers
- Tune densification schedule and learning rates

Expected improvement: Better detail, quantified quality, but same fundamental approach.

---

## Option B: Feed-Forward Gaussian Splatting (Medium-Term) ← SELECTED

**Effort:** Medium | **Impact:** High

Replace per-scene optimization with pre-trained models that predict Gaussians directly from images.

### B1: MASt3R (Replace COLMAP)
- **What:** Geometric foundation model from Naver Labs. Predicts dense 3D point clouds + camera poses from image pairs.
- **Input:** 2+ uncalibrated images
- **Speed:** Seconds (vs minutes for COLMAP)
- **License:** CC BY-NC-SA 4.0 (non-commercial)
- **Repo:** github.com/naver/mast3r
- **Use case:** Drop-in COLMAP replacement. Faster, works with fewer images, no calibration needed.

### B2: InstantSplat (NVIDIA)
- **What:** MASt3R + 3DGS in one pipeline. Sparse-view, pose-free Gaussian Splatting.
- **Input:** 2-3 images, no camera calibration needed
- **Speed:** Seconds for initial, optional refinement adds minutes
- **License:** Apache 2.0
- **Repo:** github.com/NVlabs/InstantSplat
- **Use case:** Full replacement of COLMAP + gsplat. Instant preview from a few photos.

### B3: MVSplat (ECCV 2024 Oral)
- **What:** Feed-forward model that directly predicts Gaussians from sparse multi-view images.
- **Input:** 3-6 views
- **Speed:** 22 FPS inference (real-time)
- **Quality:** State-of-the-art on RealEstate10K and ACID benchmarks
- **License:** Research
- **Repo:** github.com/donydchen/mvsplat
- **Use case:** Real-time preview as user captures images.

### B4: Splatt3R
- **What:** Built on MASt3R, predicts Gaussians from 2 uncalibrated images.
- **Input:** 2 images
- **Speed:** ~4 FPS at 512x512
- **License:** Research
- **Repo:** github.com/btsmart/splatt3r, pretrained on HuggingFace
- **Use case:** Instant 3D from just 2 photos.

### B5: AnySplat (SIGGRAPH Asia 2025)
- **What:** Feed-forward from unconstrained views. With 32+ views, matches per-scene optimization.
- **Input:** Any number of unposed images
- **Speed:** Seconds
- **Use case:** Best of both worlds — works sparse or dense.

---

## Option C: Diffusion-Enhanced Reconstruction (Future)

**Effort:** High | **Impact:** Very High

Use diffusion models as priors to fill in missing information from sparse views.

### C1: CAT3D (Google)
- **What:** Multi-view diffusion generates novel views, then reconstructs 3DGS from all views.
- **Input:** 1-6 images
- **Speed:** Minutes
- **Quality:** State-of-the-art for sparse-view, fills in unseen areas
- **Use case:** High-quality reconstruction from very few images.

### C2: ReconFusion (Google)
- **What:** View synthesis diffusion model regularizes NeRF/3DGS optimization.
- **Input:** Sparse views
- **Speed:** ~1 hour
- **Use case:** Maximum quality when time isn't critical.

### C3: DiffSplat (ICLR 2025)
- **What:** Text/image → 3D Gaussians via fine-tuned diffusion model.
- **Input:** Single image or text prompt
- **Speed:** 1-2 seconds
- **Use case:** Generate room elements, furniture, complete missing areas.

### C4: SparseGS
- **What:** 2D diffusion prior removes artifacts from sparse 3DGS reconstruction.
- **Input:** Sparse-view 3DGS + diffusion model
- **Use case:** Post-processing to clean up feed-forward results.

### C5: Generative Sparse-View GS (CVPR 2025)
- **What:** High-fidelity quality from only 3 training views using generative priors.
- **Use case:** Minimum-image room reconstruction.

---

## Option D: 3D World Generation (Long-Term)

**Effort:** Very High | **Impact:** Transformative

### D1: World Labs / Marble
- Generates navigable 3D environments from text/photos
- Exports Gaussian splats and meshes
- Commercial API, $230M funded

### D2: Tencent HunyuanWorld
- First open-source 3D world generation model
- 360-degree immersive environments
- Mesh export, object-level control

### D3: Meta WorldGen
- Text → interactive 3D worlds
- Procedural reasoning + diffusion

---

## Evaluation Framework

### Metrics
| Metric | Measures | Target |
|--------|----------|--------|
| PSNR (dB) | Pixel accuracy | 30+ |
| SSIM | Structural similarity | 0.90+ |
| LPIPS | Perceptual quality | <0.20 |
| Training time | Speed | <60s for feed-forward |
| # input images | Data efficiency | <10 |
| # Gaussians | Model size | - |

### Benchmark Datasets
| Dataset | Type | Scenes | Download |
|---------|------|--------|----------|
| Mip-NeRF 360 | Real indoor/outdoor | 9 (4 indoor) | storage.googleapis.com/gresearch/refraw360/360_v2.zip |
| DTU | Real objects | 100+ | roboimagedata.compute.dtu.dk |
| ScanNet++ | Real indoor | 460 | scan-net.org (requires agreement) |
| Replica | Synthetic indoor | ~18 | github.com/facebookresearch/Replica-Dataset |

### Published Baselines (Mip-NeRF 360 Indoor)

| Method | Kitchen PSNR | Room PSNR | Avg Indoor PSNR |
|--------|-------------|-----------|-----------------|
| 3DGS (original) | 31.14 | 31.43 | ~30.9 |
| 2DGS | 30.41 | 30.37 | ~30.0 |
| Mip-Splatting | ~31.5 | ~31.8 | ~31.2 |

---

## Implementation Plan

### Phase 1: Evaluation framework + benchmark data [DONE]
- [x] Download Mip-NeRF 360 dataset (12.5GB, 9 scenes)
- [x] Implement PSNR/SSIM/LPIPS computation (`pipeline/evaluate.py`)
- [x] Benchmark validation script (`scripts/run_benchmark_validation.py`)
- [ ] Run `--scene kitchen --sfm-backend gt` to establish trainer baseline
- [ ] Run `--scene kitchen --sfm-backend mast3r` to compare MASt3R poses

### Phase 2: MASt3R integration [DONE]
- [x] `pipeline/run_mast3r.py` — Drop-in COLMAP replacement using MASt3R from InstantSplat
- [x] Backend routing in `app/services/reconstruction.py` (sfm_backend=colmap|mast3r)
- [x] Frontend SfM selector toggle in ReconstructionSettings.tsx
- [x] Pipeline progress step handles both "Running COLMAP" and "Running MASt3R"
- [x] Schema updated (sfm_backend, mast3r_image_size fields)

### Phase 3: AnySplat evaluation [RESEARCHED]
- [x] Research: AnySplat IS available (MIT license, GitHub, HuggingFace weights)
- [x] Setup script: `scripts/setup_anysplat.py`
- [x] Pipeline module: `pipeline/run_anysplat.py`
- [ ] Install: `python3 scripts/setup_anysplat.py`
- [ ] Test on kitchen frames vs MASt3R+gsplat pipeline

**AnySplat key findings:**
- Feed-forward: single forward pass produces poses + Gaussians (no SfM needed)
- Trained on ARKitScenes (4,406 indoor scenes) + ScanNet++ (935 indoor)
- 2-200+ views, 1.4-4.1s for 32-64 views
- ~886M params, MIT license, pretrained on HuggingFace
- Caveat: needs ~16GB VRAM for 16 views at 448x448 (our GPU: 16GB)

**Alternative models researched:**
| Model | Unposed? | Views | Indoor? | Available? |
|-------|----------|-------|---------|------------|
| AnySplat | Yes | 2-200+ | Yes | Yes (MIT) |
| LongSplat | Yes | Video | Outdoor | Yes (NVIDIA) |
| NoPoSplat | Yes | 2-3 | Limited | Yes |
| VicaSplat | Yes | 2-8 | Limited | Yes |
| YoNoSplat | Yes | Many | Unknown | No (empty repo) |

### Phase 4: Streaming reconstruction [BACKEND DONE]
- [x] `pipeline/streaming.py` — Incremental reconstruction state machine
- [x] `/ws/stream/{project_id}` WebSocket endpoint for frame streaming
- [x] StreamingReconstructor: frame accumulation → MASt3R → Gaussian init → incremental refine → PLY export
- [ ] Frontend streaming capture UI (camera access, frame sending)
- [ ] Progressive 3D viewer updates during streaming

### Phase 5: Diffusion comparison (future)
- Set up CAT3D or SparseGS
- Compare against feed-forward on same benchmarks
- Evaluate sparse-view scenarios (3, 6, 12 images)
