"""Tests for editing API endpoints using FastAPI TestClient.

Uses a synthetic project with a small PLY file to test the full
request/response cycle of scene editing endpoints.
"""

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

# We need to set up the DB and create a project before tests


@pytest.fixture
def project_dir():
    """Create a temp project directory with a synthetic PLY."""
    tmp = Path(tempfile.mkdtemp(prefix="test_editing_"))

    # Create a small PLY
    from plyfile import PlyData, PlyElement

    n = 50
    dtype_fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    data = np.zeros(n, dtype=dtype_fields)
    data["x"] = np.linspace(-5, 5, n).astype(np.float32)
    data["y"] = np.zeros(n, dtype=np.float32)
    data["z"] = np.zeros(n, dtype=np.float32)
    data["opacity"] = np.ones(n, dtype=np.float32)
    data["scale_0"] = np.full(n, -3.0, dtype=np.float32)
    data["scale_1"] = np.full(n, -3.0, dtype=np.float32)
    data["scale_2"] = np.full(n, -3.0, dtype=np.float32)
    data["rot_0"] = np.ones(n, dtype=np.float32)
    data["f_dc_0"] = np.full(n, 0.5, dtype=np.float32)
    data["f_dc_1"] = np.full(n, 0.5, dtype=np.float32)
    data["f_dc_2"] = np.full(n, 0.5, dtype=np.float32)

    element = PlyElement.describe(data, "vertex")
    PlyData([element], text=False).write(str(tmp / "scene.ply"))

    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
async def client_and_project(project_dir):
    """Create a test client and a project in the DB pointing to our temp dir."""
    from app.config import settings
    from app.main import app
    from app.database import engine
    from app.models import Base, Project
    from sqlalchemy.ext.asyncio import async_sessionmaker

    # Use a temp DB
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create project
    project_id = str(uuid.uuid4())
    proj_data_dir = settings.data_dir / "projects" / project_id
    proj_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy PLY to the project dir
    shutil.copy2(project_dir / "scene.ply", proj_data_dir / "scene.ply")

    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with SessionLocal() as session:
        project = Project(
            id=project_id,
            name="test-project",
            status="ready",
        )
        session.add(project)
        await session.commit()

    # Clean scene cache
    from pipeline.scene_manager import _scene_cache, _cache_lock
    with _cache_lock:
        _scene_cache.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, project_id

    # Cleanup
    with _cache_lock:
        _scene_cache.clear()
    shutil.rmtree(proj_data_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_load_scene(client_and_project):
    client, pid = client_and_project
    resp = await client.post(f"/api/projects/{pid}/scene/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["n_gaussians"] == 50
    assert data["undo_count"] == 0
    assert data["redo_count"] == 0


@pytest.mark.asyncio
async def test_edit_delete(client_and_project):
    client, pid = client_and_project

    # Load first
    resp = await client.post(f"/api/projects/{pid}/scene/load")
    assert resp.status_code == 200

    # Delete some gaussians
    resp = await client.post(f"/api/projects/{pid}/scene/edit", json={
        "type": "delete",
        "indices": [0, 1, 2],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_affected"] == 3
    assert data["undo_count"] == 1


@pytest.mark.asyncio
async def test_edit_transform(client_and_project):
    client, pid = client_and_project

    resp = await client.post(f"/api/projects/{pid}/scene/load")
    assert resp.status_code == 200

    resp = await client.post(f"/api/projects/{pid}/scene/edit", json={
        "type": "transform",
        "indices": [0, 1, 2],
        "translation": [1.0, 0.0, 0.0],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_affected"] == 3
    assert "translate" in data["label"]


@pytest.mark.asyncio
async def test_undo_redo(client_and_project):
    client, pid = client_and_project

    # Load + edit
    await client.post(f"/api/projects/{pid}/scene/load")
    await client.post(f"/api/projects/{pid}/scene/edit", json={
        "type": "delete",
        "indices": [5, 6],
    })

    # Undo
    resp = await client.post(f"/api/projects/{pid}/scene/undo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["undo_count"] == 0
    assert data["redo_count"] == 1

    # Redo
    resp = await client.post(f"/api/projects/{pid}/scene/redo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["undo_count"] == 1
    assert data["redo_count"] == 0


@pytest.mark.asyncio
async def test_undo_nothing(client_and_project):
    client, pid = client_and_project
    await client.post(f"/api/projects/{pid}/scene/load")

    resp = await client.post(f"/api/projects/{pid}/scene/undo")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_history(client_and_project):
    client, pid = client_and_project

    await client.post(f"/api/projects/{pid}/scene/load")
    await client.post(f"/api/projects/{pid}/scene/edit", json={
        "type": "delete", "indices": [0],
    })
    await client.post(f"/api/projects/{pid}/scene/edit", json={
        "type": "delete", "indices": [1],
    })

    resp = await client.get(f"/api/projects/{pid}/scene/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["undo_count"] == 2
    assert data["dirty"] is True


@pytest.mark.asyncio
async def test_query_box(client_and_project):
    client, pid = client_and_project
    await client.post(f"/api/projects/{pid}/scene/load")

    resp = await client.post(f"/api/projects/{pid}/scene/query-box", json={
        "min": [-1.0, -1.0, -1.0],
        "max": [1.0, 1.0, 1.0],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    assert len(data["indices"]) == data["count"]


@pytest.mark.asyncio
async def test_query_sphere(client_and_project):
    client, pid = client_and_project
    await client.post(f"/api/projects/{pid}/scene/load")

    resp = await client.post(f"/api/projects/{pid}/scene/query-sphere", json={
        "center": [0.0, 0.0, 0.0],
        "radius": 2.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0


@pytest.mark.asyncio
async def test_delete_region_box_inside(client_and_project):
    client, pid = client_and_project
    await client.post(f"/api/projects/{pid}/scene/load")

    resp = await client.post(f"/api/projects/{pid}/scene/delete-region", json={
        "shape": "box",
        "min": [-1.0, -1.0, -1.0],
        "max": [1.0, 1.0, 1.0],
        "mode": "inside",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_affected"] > 0


@pytest.mark.asyncio
async def test_delete_region_sphere_outside(client_and_project):
    client, pid = client_and_project
    await client.post(f"/api/projects/{pid}/scene/load")

    resp = await client.post(f"/api/projects/{pid}/scene/delete-region", json={
        "shape": "sphere",
        "center": [0.0, 0.0, 0.0],
        "radius": 1.0,
        "mode": "outside",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_affected"] > 0


@pytest.mark.asyncio
async def test_edit_no_indices(client_and_project):
    client, pid = client_and_project
    await client.post(f"/api/projects/{pid}/scene/load")

    resp = await client.post(f"/api/projects/{pid}/scene/edit", json={
        "type": "delete",
        "indices": [],
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_project_not_found(client_and_project):
    client, _ = client_and_project

    resp = await client.post("/api/projects/nonexistent/scene/load")
    assert resp.status_code == 404
