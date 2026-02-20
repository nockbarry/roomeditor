import os
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings


def _find_ffmpeg() -> str:
    """Find ffmpeg binary, preferring local install."""
    local = Path.home() / ".local" / "bin" / "ffmpeg"
    if local.exists():
        return str(local)
    system = shutil.which("ffmpeg")
    if system:
        return system
    return "ffmpeg"


def _setup_cuda():
    """Set CUDA_HOME if a local CUDA toolkit exists."""
    cuda_home = Path.home() / ".local" / "cuda-12.8"
    if cuda_home.exists() and "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["PATH"] = str(cuda_home / "bin") + ":" + os.environ.get("PATH", "")


_setup_cuda()


class Settings(BaseSettings):
    app_name: str = "Room Editor"
    data_dir: Path = Path(__file__).parent.parent / "data"
    database_url: str = "sqlite+aiosqlite:///./roomeditor.db"
    max_upload_size_mb: int = 2048
    ffmpeg_path: str = _find_ffmpeg()
    ffmpeg_fps: int = 2
    gsplat_iterations: int = 15_000
    gsplat_sh_degree: int = 2
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_prefix": "ROOMEDITOR_"}


settings = Settings()

# Ensure data directory exists
settings.data_dir.mkdir(parents=True, exist_ok=True)
