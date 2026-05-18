"""Pydantic models for request/response and the camera-placement enum.

Values are imported directly from api_engine/config.py — api_engine/config.py
is the single source of truth for placement strings.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from api_engine.config import (
    PLACEMENT_OUTSIDE_FACING_ENTRANCE,
    PLACEMENT_INSIDE_FACING_EXIT,
    PLACEMENT_INSIDE_EXIT_ON_RIGHT,
    PLACEMENT_INSIDE_EXIT_ON_LEFT,
    PLACEMENT_INSIDE_EXIT_ON_BOTH,
)


class CameraPlacement(str, Enum):
    OUTSIDE_FACING_ENTRANCE = PLACEMENT_OUTSIDE_FACING_ENTRANCE
    INSIDE_FACING_EXIT = PLACEMENT_INSIDE_FACING_EXIT
    INSIDE_EXIT_ON_RIGHT = PLACEMENT_INSIDE_EXIT_ON_RIGHT
    INSIDE_EXIT_ON_LEFT = PLACEMENT_INSIDE_EXIT_ON_LEFT
    INSIDE_EXIT_ON_BOTH_SIDES = PLACEMENT_INSIDE_EXIT_ON_BOTH


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    camera_placement: str
    json_every_n_frames: int
    classify_every_n_frames: int
    pops_classify_every_n_frames: int
    include_annotated_video: bool = False
    queued_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    result_url: Optional[str] = Field(
        default=None,
        description="Set only when status is 'succeeded' — relative URL to the JSON.",
    )
    annotated_video_url: Optional[str] = Field(
        default=None,
        description=(
            "Relative URL to the annotated MP4. Only present when the job was "
            "created with include_annotated_video=true and the encode succeeded."
        ),
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    device: str
    engine_ready: bool
