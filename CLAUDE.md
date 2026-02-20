# Room Editor

3D room reconstruction and editing application using Gaussian Splatting.

## Project Structure

- `server/` — Python/FastAPI backend with GPU pipeline
- `client/` — React/TypeScript frontend with Three.js 3D viewer

## Development

### Server
```bash
cd server
uv sync --python /usr/bin/python3  # Must use system Python 3.10 (has pycolmap, torch, gsplat)
export CUDA_HOME=$HOME/.local/cuda-12.8
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Client
```bash
cd client
npm install
npm run dev
```

### Client
```bash
cd client
npm install
npm run dev
```

## Key Conventions

- Backend uses FastAPI with async SQLAlchemy + SQLite
- Frontend uses React 19 + TypeScript + Vite + Tailwind CSS
- 3D rendering uses Three.js + GaussianSplats3D
- State management with Zustand
- GPU pipeline steps are in `server/pipeline/`
- Project data stored in `server/data/projects/{uuid}/`

## Environment

- System Python 3.10 has pycolmap 3.13, torch 2.9.1+cu128, gsplat 1.5.3 installed globally
- The uv venv links to system packages via a .pth file
- CUDA toolkit at `~/.local/cuda-12.8` (required for gsplat JIT compilation)
- ffmpeg at `~/.local/bin/ffmpeg`
- GPU: NVIDIA RTX 4070 Ti SUPER (16GB VRAM)

## pycolmap 3.13 API Notes

- `image.cam_from_world()` is a **method call** (needs parentheses), not a property
- `pycolmap.extract_features()` uses `extraction_options=` (FeatureExtractionOptions)
- `pycolmap.match_exhaustive()` uses `matching_options=` (FeatureMatchingOptions)
- All path arguments must be `str()`, not `Path` objects
- `incremental_mapping()` returns `dict[int, Reconstruction]`

## gsplat 1.5.3 Notes

- Requires `info["means2d"].retain_grad()` before `loss.backward()` for densification
- Needs CUDA toolkit (nvcc) for JIT compilation — set CUDA_HOME
- `export_splats(format="ply")` produces standard 3DGS PLY format
