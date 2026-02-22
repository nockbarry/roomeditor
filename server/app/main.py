import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine
from app.models import Base
from app.routes import anysplat, benchmarks, collab, jobs, objects, postprocess, projects, segments, ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(projects.router)
app.include_router(anysplat.router)
app.include_router(segments.router)
app.include_router(postprocess.router)
app.include_router(objects.router)
app.include_router(jobs.router)
app.include_router(ws.router)
app.include_router(collab.router)
app.include_router(benchmarks.router)

# Serve benchmark data files (renders, etc.) — must be before /data mount
data_benchmarks_dir = settings.data_dir / "benchmarks"
data_benchmarks_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data/benchmarks", StaticFiles(directory=str(data_benchmarks_dir)), name="benchmark_data")

# Serve project data files (PLY, segments, etc.)
data_projects_dir = settings.data_dir / "projects"
data_projects_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(data_projects_dir)), name="project_data")


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": settings.app_name}
