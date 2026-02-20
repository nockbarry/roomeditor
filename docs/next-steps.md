# Next Steps — Room Editor / AnySplat Studio

## What Was Completed

### AnySplat Studio (3-panel workspace)
- Left: CompactUploader + FrameGallery with sharpness indicators, shift-click range selection, subsampling visualization
- Center: SplatViewer (GaussianSplats3D) that reloads on each rebuild
- Right: Tabbed panel — Settings (rebuild) + Objects (SAM2 segments)
- Variable FPS extraction (1-10fps), 4 presets (Quick/Standard/High/Chunked)
- VRAM and time estimates displayed before rebuild

### Post-Processing Pipeline
- **Floater pruning** endpoint + UI — removes low-opacity gaussians from PLY
- **Training refinement** endpoint + UI — loads AnySplat PLY + cameras.json, runs 2000 iterations of 3DGS training via gsplat GaussianTrainer
- **Quality stats** endpoint + UI — opacity distribution, scale stats, spatial density
- All three auto-chain: rebuild → auto-fetch stats, prune/refine → re-fetch stats

### AnySplat Enhancements
- Full SH export (degree 4, 25 coefficients) via `--full_sh` flag
- Camera intrinsics export (fx, fy, cx, cy per view) in cameras.json
- Chunked mode exposed in UI (chunk_size, chunk_overlap)
- Frame subsampling logic mirrored client-side to show which frames will actually be used

### SAM2 Segmentation
- Click-to-segment on frames, auto-propagate masks across video
- Segment list panel with color-coded masks
- Backend: SAM2 loaded via PYTHONPATH, checkpoint at sam2.1_hiera_small.pt

---

## Needs Testing

These features were implemented but haven't been fully tested end-to-end:

1. **Chunked mode** — Does the PLY merging from run_inference_max.py produce good results? Test with a video that has >64 frames selected, chunk_size=32, overlap=8.

2. **Training refinement** — The `_run_refine()` function in `server/app/routes/postprocess.py` loads cameras.json and initializes gsplat GaussianTrainer. Needs testing:
   - Does the c2w → viewmat conversion produce correct poses?
   - Do the normalized → pixel intrinsics convert correctly?
   - Does the opacity logit conversion (AnySplat raw [0,1] → gsplat logits) work?
   - Does 2000 iterations actually improve quality, or does it diverge?

3. **Full SH export** — Verify that the viewer (GaussianSplats3D) renders view-dependent color correctly from 25 SH coefficients. Some viewers only support SH degree 0 (DC only).

4. **Floater pruning** — Test with real PLY. Verify opacity_threshold=0.01 is a good default. Check that the PLY written by plyfile is loadable by the viewer.

5. **Quality stats** — The `compute_gaussian_stats()` function from evaluate_noreference.py needs to be verified. Make sure the import path and function signature are correct.

---

## Next Features to Build

### High Priority

1. **MASt3R depth-guided refinement**
   - Already have MASt3R in the codebase (`server/pipeline/run_mast3r.py`)
   - MASt3R produces dense depth maps from image pairs
   - Could use as depth supervision during training refinement
   - Would significantly improve geometry quality

2. **TSDF mesh extraction**
   - `server/pipeline/extract_mesh.py` already exists with `extract_mesh_tsdf()`
   - After refinement, extract a triangle mesh from the gaussian field
   - Useful for physics, collision, and room editing operations
   - Add "Extract Mesh" button in post-process section

3. **Real-time streaming reconstruction**
   - Instead of waiting for all frames → rebuild, show progressive results
   - AnySplat is fast enough (3-10s) to rebuild on each new batch
   - Upload a few frames → get initial splat → upload more → rebuild with all
   - Would make the workflow feel very interactive

4. **Gaussian editing tools**
   - Select region in 3D viewer → delete gaussians (remove objects)
   - Combine with SAM2 masks: select object in 2D → find corresponding 3D gaussians
   - Scale/rotate/translate groups of gaussians
   - This is the "room editing" in "room editor"

### Medium Priority

5. **Multi-video support**
   - Upload multiple videos/photo sets for the same room
   - Merge frame sets, deduplicate similar views
   - Better coverage from different angles

6. **PLY compression / optimization**
   - Current PLYs can be 50-200MB
   - Implement gaussian count reduction (merge similar nearby gaussians)
   - Quantization of SH coefficients
   - Could use the existing compress_splat.py as starting point

7. **Camera path export**
   - Export cameras.json as a smooth interpolated camera path
   - Render fly-through videos from the gaussian field
   - Useful for presentations / walkthroughs

8. **Comparison view**
   - Side-by-side or A/B toggle between original AnySplat output vs refined
   - Before/after pruning comparison
   - Different preset results comparison

### Lower Priority

9. **Object generation / insertion**
   - Generate 3D objects (furniture, decorations) to place in the scene
   - Would need a 3D generation model (e.g., InstantMesh, DreamGaussian)
   - Place at specific world coordinates in the gaussian field

10. **Room structure editing**
    - Detect walls/floors/ceilings from the gaussian field or mesh
    - Allow resizing rooms, moving walls
    - This is very ambitious but is the end goal

---

## Known Issues / Technical Debt

- **Training refinement hardcodes 2000 iterations** — should be configurable and show progress
- **No progress feedback during refinement** — it can take 30-60s, user sees no feedback beyond spinner
- **AnySplat venv path is hardcoded** in run_anysplat_subprocess.py
- **SAM2 checkpoint path is hardcoded** in server/pipeline/run_sam2.py
- **No error recovery** — if refinement or pruning fails, the PLY could be corrupted (backup exists but no auto-restore)
- **VRAM estimates are rough** — based on heuristics, not measured. Should profile actual usage.
- **Camera intrinsics conversion** — the normalized → pixel conversion in run_inference_max.py needs verification with real AnySplat output

---

## File Reference

### Key files to understand the codebase:
- `server/app/routes/anysplat.py` — extract-frames, frame selection, anysplat-run endpoints
- `server/app/routes/postprocess.py` — prune, refine, quality-stats endpoints
- `server/pipeline/run_anysplat_subprocess.py` — AnySplat subprocess wrapper
- `server/pipeline/select_keyframes.py` — frame scoring and selection
- `server/pipeline/compress_splat.py` — PLY pruning implementation
- `client/src/stores/anysplatStore.ts` — all state + actions for the studio
- `client/src/stores/segmentStore.ts` — SAM2 segmentation state
- `client/src/pages/AnySplatStudio.tsx` — main studio layout
- `client/src/components/studio/` — all studio panel components
- `/home/nock/projects/anysplat/run_inference_max.py` — modified AnySplat inference script
