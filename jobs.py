"""In-memory job store. Thread-safe dict-backed.

A single Cloud Run instance (--max-instances=1) is enough for v1, so this is
correct and simple. To scale horizontally later, swap this class for a
Firestore / Redis / Cloud Tasks implementation behind the same surface — the
methods used by main.py and worker.py (`create`, `get`, `update`, `delete`,
`all`) are the contract.
"""
from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from schemas import JobStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    camera_placement: str
    json_every_n_frames: int
    classify_every_n_frames: int
    pops_classify_every_n_frames: int
    video_path: Path
    work_dir: Path
    include_annotated_video: bool = False
    result_path: Optional[Path] = None
    annotated_video_path: Optional[Path] = None
    error: Optional[str] = None
    queued_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class InMemoryJobStore:
    """Thread-safe dict-backed store.

    When this is swapped for a persistent backend (Firestore / Redis), add a
    startup reconciliation pass that marks any RUNNING job as FAILED. The
    in-memory store dies with the process so the orphan-RUNNING risk is
    moot today, but it becomes real the moment state survives a restart.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self, *, camera_placement: str, json_every_n_frames: int,
               classify_every_n_frames: int, pops_classify_every_n_frames: int,
               video_path: Path, work_dir: Path,
               include_annotated_video: bool = False,
               job_id: Optional[str] = None) -> JobRecord:
        job = JobRecord(
            job_id=job_id or uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            camera_placement=camera_placement,
            json_every_n_frames=json_every_n_frames,
            classify_every_n_frames=classify_every_n_frames,
            pops_classify_every_n_frames=pops_classify_every_n_frames,
            video_path=video_path,
            work_dir=work_dir,
            include_annotated_video=include_annotated_video,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            return job

    def update_if_not_cancelled(self, job_id: str, **fields) -> Optional[JobRecord]:
        # CANCELLED wins over any later terminal state — the worker uses this to
        # avoid stomping a DELETE that arrived during inference.
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == JobStatus.CANCELLED:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def all(self) -> Iterator[JobRecord]:
        with self._lock:
            return iter(list(self._jobs.values()))


def gc_expired(store: InMemoryJobStore, ttl_minutes: int) -> int:
    """Delete terminal jobs older than ttl_minutes and their on-disk artifacts.

    Returns count of jobs evicted.
    """
    cutoff = utcnow() - timedelta(minutes=ttl_minutes)
    evicted = 0
    for job in list(store.all()):
        if not job.is_terminal:
            continue
        finished = job.finished_at or job.queued_at
        if finished <= cutoff:
            shutil.rmtree(job.work_dir, ignore_errors=True)
            store.delete(job.job_id)
            evicted += 1
    return evicted
