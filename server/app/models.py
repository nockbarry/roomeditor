import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="created"
    )  # created, uploading, processing, ready, error
    video_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    gaussian_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reconstruction_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # reconstruct, segment, generate, inpaint
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # pending, running, completed, failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 to 1.0
    current_step: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class SceneObject(Base):
    __tablename__ = "scene_objects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    segment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(255), default="unknown")
    gaussian_start: Mapped[int] = mapped_column(Integer, default=0)
    gaussian_count: Mapped[int] = mapped_column(Integer, default=0)
    centroid: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # [x, y, z]
    bbox_min: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # [x, y, z]
    bbox_max: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # [x, y, z]
    translation: Mapped[dict] = mapped_column(JSON, default=lambda: [0.0, 0.0, 0.0])
    rotation: Mapped[dict] = mapped_column(JSON, default=lambda: [0.0, 0.0, 0.0, 1.0])
    scale: Mapped[dict] = mapped_column(JSON, default=lambda: [1.0, 1.0, 1.0])
    visible: Mapped[bool] = mapped_column(Boolean, default=True)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(
        String(50), default="reconstruction"
    )  # reconstruction, generated, imported
    source_file: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
