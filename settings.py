"""Runtime configuration via environment variables.

All knobs are POPS_API_* prefixed so they don't collide with the existing
api/main.py service or the Gradio demo. Defaults are production-safe for a
single-instance Cloud Run service.
"""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # The HTTP port is read directly from $PORT by uvicorn (Cloud Run sets it),
    # so we don't replicate it as a Settings field.
    model_config = SettingsConfigDict(env_prefix="POPS_API_", case_sensitive=False)

    log_level: str = "INFO"

    max_upload_mb: int = Field(default=500, gt=0)
    max_queue: int = Field(default=4, gt=0)
    job_ttl_min: int = Field(default=60, gt=0)

   
    classify_every_n_frames: int = Field(default=1, gt=0)
    pops_classify_every_n_frames: int = Field(default=8, gt=0)

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    allowed_video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
