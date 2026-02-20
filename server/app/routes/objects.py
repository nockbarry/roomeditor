from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import SceneObject
from app.schemas import ObjectTransformUpdate, SceneObjectResponse

router = APIRouter(prefix="/api/projects/{project_id}/objects", tags=["objects"])


@router.get("", response_model=list[SceneObjectResponse])
async def list_objects(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SceneObject)
        .where(SceneObject.project_id == project_id)
        .order_by(SceneObject.segment_id)
    )
    return result.scalars().all()


@router.get("/{object_id}", response_model=SceneObjectResponse)
async def get_object(project_id: str, object_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SceneObject).where(
            SceneObject.id == object_id, SceneObject.project_id == project_id
        )
    )
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")
    return obj


@router.put("/{object_id}/transform", response_model=SceneObjectResponse)
async def update_transform(
    project_id: str,
    object_id: str,
    data: ObjectTransformUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SceneObject).where(
            SceneObject.id == object_id, SceneObject.project_id == project_id
        )
    )
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    if data.translation is not None:
        obj.translation = data.translation
    if data.rotation is not None:
        obj.rotation = data.rotation
    if data.scale is not None:
        obj.scale = data.scale
    if data.visible is not None:
        obj.visible = data.visible
    if data.locked is not None:
        obj.locked = data.locked

    await db.commit()
    await db.refresh(obj)
    return obj


@router.delete("/{object_id}", status_code=204)
async def delete_object(project_id: str, object_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SceneObject).where(
            SceneObject.id == object_id, SceneObject.project_id == project_id
        )
    )
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    obj.visible = False
    await db.commit()
