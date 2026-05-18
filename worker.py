"""Background worker — drains the job queue and runs the headless engine.

All CUDA inference is pinned to a single OS thread via the
ThreadPoolExecutor(max_workers=1) so we don't pay per-request CUDA stream
synchronisation. Each job produces a dict (the _tracking.json structure),
which the worker serialises into the job's work dir.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from jobs import InMemoryJobStore, JobRecord, gc_expired, utcnow
from logging_config import bind_job_id, get_logger
from schemas import JobStatus

if TYPE_CHECKING:
    from api_engine import HeadlessTrackingEngine

log = get_logger(__name__)


def _sanitise_floats(obj):
    """Replace NaN/Inf floats with None so the output is valid JSON.

    json.dump's default allow_nan=True emits literal NaN/Infinity tokens
    that strict parsers (browsers, BigQuery) reject. Walk the result and
    swap them for None before serialising.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitise_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_floats(v) for v in obj]
    return obj


def _run_one(engine: "HeadlessTrackingEngine", job: JobRecord) -> tuple[Path, Optional[Path]]:
    """Run the engine end-to-end and write tracking.json into the job's work_dir.

    Returns (json_path, annotated_video_path-or-None). Runs on the executor
    thread, so blocking calls are fine here.
    """
    try:
        video_out = (
            str(job.work_dir / "annotated.mp4")
            if job.include_annotated_video else None
        )
        result = engine.process_video(
            str(job.video_path),
            camera_placement=job.camera_placement,
            json_every_n_frames=job.json_every_n_frames,
            classify_every_n_frames=job.classify_every_n_frames,
            pops_classify_every_n_frames=job.pops_classify_every_n_frames,
            annotated_video_path=video_out,
        )
        result = _sanitise_floats(result)
        dest = job.work_dir / "tracking.json"
        tmp = dest.with_suffix(".json.tmp")
        # allow_nan=False is belt-and-suspenders: if anything slipped past
        # _sanitise_floats it surfaces as a FAILED job, not invalid JSON
        # served to the client.
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        os.replace(tmp, dest)
        produced = result.get("processing_info", {}).get("annotated_video_path")
        video_path = Path(produced) if produced else None
        return dest, video_path
    finally:
        # Free the per-job caching allocator slabs so repeated inference on
        # a long-lived instance doesn't fragment VRAM into eventual OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Worker:
    """Pulls job IDs off an asyncio.Queue and runs them serially on the executor."""

    def __init__(self, *, store: InMemoryJobStore, engine: "HeadlessTrackingEngine",
                 executor: ThreadPoolExecutor, queue: asyncio.Queue[str]):
        self.store = store
        self.engine = engine
        self.executor = executor
        self.queue = queue
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        log.info("worker.start")
        while not self._stop.is_set():
            try:
                job_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                # An exception out here (store, queue, network) would kill
                # the loop silently before; log loudly and keep going so the
                # next job has a chance to drain.
                log.exception("worker.queue_get_failed")
                continue
            try:
                await self._handle(job_id)
            except Exception:
                log.exception("worker.handle_unexpected", job_id=job_id)
        log.info("worker.stop")

    async def _handle(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            log.warning("worker.unknown_job", job_id=job_id)
            return
        if job.status == JobStatus.CANCELLED:
            log.info("worker.skip_cancelled", job_id=job_id)
            return

        bind_job_id(job_id)
        self.store.update(job_id, status=JobStatus.RUNNING, started_at=utcnow())
        log.info("worker.job_start", camera_placement=job.camera_placement,
                 json_every_n=job.json_every_n_frames,
                 classify_every_n=job.classify_every_n_frames)

        loop = asyncio.get_running_loop()
        try:
            result_path, video_path = await loop.run_in_executor(
                self.executor, _run_one, self.engine, job)
            updated = self.store.update_if_not_cancelled(
                job_id,
                status=JobStatus.SUCCEEDED,
                result_path=result_path,
                annotated_video_path=video_path,
                finished_at=utcnow(),
            )
            if updated is None:
                # Cancelled during inference. Discard the result and the work_dir
                # so we don't ship artifacts the client asked us to abandon.
                shutil.rmtree(job.work_dir, ignore_errors=True)
                log.info("worker.completed_after_cancel", job_id=job_id)
            else:
                log.info("worker.job_succeeded", result_path=str(result_path))
        except Exception as exc:
            tb = traceback.format_exc()
            updated = self.store.update_if_not_cancelled(
                job_id,
                status=JobStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}\n{tb}",
                finished_at=utcnow(),
            )
            if updated is None:
                shutil.rmtree(job.work_dir, ignore_errors=True)
                log.info("worker.failed_after_cancel", job_id=job_id)
            else:
                log.exception("worker.job_failed", error=str(exc))
        finally:
            bind_job_id(None)

    def stop(self) -> None:
        self._stop.set()


async def gc_loop(store: InMemoryJobStore, ttl_minutes: int, interval_seconds: int = 60):
    """Periodically GC terminal jobs older than ttl_minutes."""
    while True:
        try:
            evicted = gc_expired(store, ttl_minutes)
            if evicted:
                log.info("gc.evicted", count=evicted)
        except Exception:
            log.exception("gc.error")
        await asyncio.sleep(interval_seconds)
