from datetime import datetime

from pydantic import BaseModel


# --- Projects ---


class ProjectCreate(BaseModel):
    name: str


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: str
    video_filename: str | None = None
    gaussian_count: int | None = None
    reconstruction_mode: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- AnySplat Studio ---


class FrameInfo(BaseModel):
    filename: str
    source_file: str
    source_type: str = ""
    sharpness: float
    selected: bool


class FrameManifest(BaseModel):
    frames: list[FrameInfo]
    total: int
    selected_count: int


class ExtractFramesRequest(BaseModel):
    fps: int = 2


class AnySplatRunRequest(BaseModel):
    max_views: int = 32
    resolution: int = 0
    chunked: bool = False
    chunk_size: int = 32
    chunk_overlap: int = 8


class AnySplatRunResult(BaseModel):
    status: str
    n_gaussians: int
    duration_sec: float
    resolution_used: int
    views_used: int
    ply_url: str
    cleanup_stats: dict | None = None


# --- Jobs ---


class JobResponse(BaseModel):
    id: str
    project_id: str
    job_type: str
    status: str
    progress: float
    current_step: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobProgress(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: str | None = None
    error_message: str | None = None


# --- Scene Objects ---


class SceneObjectResponse(BaseModel):
    id: str
    project_id: str
    segment_id: int
    label: str
    gaussian_start: int
    gaussian_count: int
    centroid: list[float] | None = None
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    translation: list[float]
    rotation: list[float]
    scale: list[float]
    visible: bool
    locked: bool
    source: str

    model_config = {"from_attributes": True}


class ObjectTransformUpdate(BaseModel):
    translation: list[float] | None = None
    rotation: list[float] | None = None
    scale: list[float] | None = None
    visible: bool | None = None
    locked: bool | None = None


class TrainingConfig(BaseModel):
    iterations: int = 15_000
    sh_degree: int = 2
    mode: str = "3dgs"
    sfm_backend: str = "colmap"  # "colmap", "mast3r", "anysplat", or "spfsplat"
    mast3r_image_size: int = 512
    spfsplat_max_views: int = 8
    depth_reg_weight: float = 0.1
    opacity_reg_weight: float = 0.0
    scale_reg_weight: float = 0.0
    flatten_reg_weight: float = 0.0
    distortion_weight: float = 0.0
    normal_weight: float = 0.0
    prune_opa: float = 0.005
    densify_until_pct: float = 0.5
    appearance_embeddings: bool = False
    tidi_pruning: bool = False
    resume_from: str | None = None
    anysplat_max_views: int = 32
    anysplat_chunked: bool = False

    model_config = {"extra": "ignore"}


# --- Segmentation ---


class SegmentInfo(BaseModel):
    id: int
    label: str
    area: int
    bbox: list[int]
    confidence: float
    primary_frame: str
    color: list[int]
    n_gaussians: int = 0
    click_point: list[int] | None = None
    visible: bool = True
    semantic_confidence: float | None = None


class SegmentManifest(BaseModel):
    segments: list[SegmentInfo]
    total: int
    primary_frame: str | None = None
    total_gaussians: int = 0
    unassigned_gaussians: int = 0


class ClickSegmentRequest(BaseModel):
    frame: str
    x: int
    y: int


class SegmentTransformRequest(BaseModel):
    translation: list[float] | None = None
    rotation: list[float] | None = None
    scale: list[float] | None = None


class VisibilityRequest(BaseModel):
    visible: bool


class DuplicateRequest(BaseModel):
    offset: list[float] = [0.5, 0.0, 0.0]


class RenameRequest(BaseModel):
    label: str


class BatchTransformRequest(BaseModel):
    segment_ids: list[int]
    transform: SegmentTransformRequest


class MergeSegmentsRequest(BaseModel):
    segment_ids: list[int]
    label: str | None = None


class SplitSegmentRequest(BaseModel):
    n_clusters: int = 2


class LightingRequest(BaseModel):
    brightness: float = 1.0
    color_tint: list[float] = [1.0, 1.0, 1.0]
    sh_scale: float = 1.0


class GenerateObjectRequest(BaseModel):
    prompt: str | None = None  # For text-to-3D
    # image uploaded separately via multipart
