"""FastAPI app for the pops_api service.

Endpoints (all under /v1):
    POST   /v1/jobs                 multipart upload, returns {job_id}
    GET    /v1/jobs/{job_id}        status (queued/running/succeeded/failed/cancelled)
    GET    /v1/jobs/{job_id}/result streams the _tracking.json
    DELETE /v1/jobs/{job_id}        cancels and removes the job
    GET    /v1/healthz              liveness
    GET    /v1/readyz               readiness (after engine warm-up)

Run locally:
    cd pops_api_v1
    python -m uvicorn main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api_engine import HeadlessTrackingEngine

__version__ = "0.1.0"
from jobs import InMemoryJobStore
from logging_config import RequestIdMiddleware, configure_logging, get_logger
from schemas import (
    CameraPlacement,
    HealthResponse,
    JobCreatedResponse,
    JobStatus,
    JobStatusResponse,
)
from settings import get_settings
from worker import Worker, gc_loop

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("pops_api")

_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "pops_api_jobs"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup.begin", version=__version__)
    # Single engine instance shared across requests; CUDA work serialised on
    # a single-thread executor so we don't pay per-request stream sync.
    engine = HeadlessTrackingEngine(device="auto")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pops-infer")
    # Separate pool for upload disk-writes and cv2 header parsing. Sized to
    # the Cloud Run concurrency limit (8) so simultaneous uploads never block
    # the event loop or starve the inference executor.
    io_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pops-io")
    store = InMemoryJobStore()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.max_queue)

    worker = Worker(store=store, engine=engine, executor=executor, queue=queue)
    worker_task = asyncio.create_task(worker.run_forever(), name="pops-worker")
    gc_task = asyncio.create_task(
        gc_loop(store, settings.job_ttl_min), name="pops-gc")

    app.state.engine = engine
    app.state.store = store
    app.state.queue = queue
    app.state.io_executor = io_executor
    app.state.engine_ready = False

    log.info("startup.engine_constructed", device=engine.device)
    # Warmup runs one synthetic forward pass on the inference executor so
    # /readyz only goes green once the first real request won't pay the
    # cold-start tax. Queued jobs (if any) serialise behind it because the
    # executor is single-thread.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, engine.warmup)
    app.state.engine_ready = True

    log.info("startup.complete", device=engine.device)
    try:
        yield
    finally:
        log.info("shutdown.begin")
        worker.stop()
        gc_task.cancel()
        worker_task.cancel()
        for t in (worker_task, gc_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        executor.shutdown(wait=False, cancel_futures=True)
        io_executor.shutdown(wait=False, cancel_futures=True)
        log.info("shutdown.complete")


app = FastAPI(
    title="POPS Video : Tracking JSON API",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)


def _validate_video(upload: UploadFile) -> str:
    """Return the validated, lowercased extension (e.g. '.mp4'). 415 on mismatch."""
    name = (upload.filename or "").lower()
    ext = os.path.splitext(name)[1]
    if ext not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video extension {ext!r}. Allowed: "
                   f"{', '.join(settings.allowed_video_extensions)}",
        )
    return ext


def _verify_playable(video_path: Path) -> None:
    """Open the file with cv2 and fail with 415 if it can't be decoded.

    Cheap content-shape check that catches text/image/random-binary uploads
    renamed to .mp4 before we hand them to the engine. Not a security
    boundary — just a sanity filter.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
            raise HTTPException(
                status_code=415,
                detail="Uploaded file is not a readable video.",
            )
    finally:
        cap.release()


async def _save_upload(
    upload: UploadFile,
    dest: Path,
    max_bytes: int,
    loop: asyncio.AbstractEventLoop,
    io_executor: ThreadPoolExecutor,
) -> int:
    """Stream the upload to disk without blocking the event loop.

    Each chunk read is already async (Starlette); the write is offloaded to
    the I/O executor so concurrent uploads don't serialize disk syscalls on
    the loop thread.
    """
    written = 0
    chunk_size = 1024 * 1024
    too_large = False
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                too_large = True
                break
            await loop.run_in_executor(io_executor, f.write, chunk)
    if too_large:
        # On Windows the file must be closed before unlink — hence the
        # break-out-of-with-then-cleanup shape.
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds max size of {max_bytes // (1024*1024)} MB",
        )
    return written


@app.get("/v1/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    engine: Optional[HeadlessTrackingEngine] = getattr(app.state, "engine", None)
    return HealthResponse(
        version=__version__,
        device=engine.device if engine else "unknown",
        engine_ready=bool(getattr(app.state, "engine_ready", False)),
    )


@app.get("/v1/readyz")
async def readyz():
    if not getattr(app.state, "engine_ready", False):
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True}


@app.post("/v1/jobs", response_model=JobCreatedResponse, status_code=202)
async def create_job(
    video: UploadFile = File(..., description="Video file (mp4/avi/mov/mkv)"),
    camera_placement: CameraPlacement = Form(
        CameraPlacement.OUTSIDE_FACING_ENTRANCE,
        description="One of the five legal placement strings the engine understands.",
    ),
    json_every_n_frames: int = Form(
        1, ge=1, le=60,
        description="Sample one JSON frame every N processed frames (1 = every frame).",
    ),
    include_annotated_video: bool = Form(
        False,
        description=(
            "When true the engine also writes a Gradio-style annotated MP4 "
            "(bboxes, trails, fill/bag/POPS chips, link lines, HUD). The "
            "URL is returned in the job status as `annotated_video_url`."
        ),
    ),
):
    ext = _validate_video(video)

    store = app.state.store
    queue: asyncio.Queue[str] = app.state.queue

    if queue.full():
        raise HTTPException(
            status_code=429,
            detail=f"Queue full ({settings.max_queue} jobs in flight). Retry shortly.",
        )

    job_id = uuid.uuid4().hex
    work_dir = _UPLOAD_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / f"input{ext}"

    max_bytes = settings.max_upload_mb * 1024 * 1024
    loop = asyncio.get_running_loop()
    io_executor: ThreadPoolExecutor = app.state.io_executor
    try:
        await _save_upload(video, video_path, max_bytes, loop, io_executor)
        await loop.run_in_executor(io_executor, _verify_playable, video_path)
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    job = store.create(
        job_id=job_id,
        camera_placement=camera_placement.value,
        json_every_n_frames=json_every_n_frames,
        classify_every_n_frames=settings.classify_every_n_frames,
        pops_classify_every_n_frames=settings.pops_classify_every_n_frames,
        video_path=video_path,
        work_dir=work_dir,
        include_annotated_video=include_annotated_video,
    )

    await queue.put(job.job_id)
    log.info("api.job_created", job_id=job.job_id,
             camera_placement=camera_placement.value,
             json_every_n=json_every_n_frames,
             include_annotated_video=include_annotated_video)
    return JobCreatedResponse(job_id=job.job_id, status=JobStatus.QUEUED)


def _job_to_status(job) -> JobStatusResponse:
    succeeded = job.status == JobStatus.SUCCEEDED
    result_url = (
        f"/v1/jobs/{job.job_id}/result"
        if succeeded and job.result_path
        else None
    )
    annotated_video_url = (
        f"/v1/jobs/{job.job_id}/annotated_video"
        if succeeded and job.annotated_video_path
        else None
    )
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        camera_placement=job.camera_placement,
        json_every_n_frames=job.json_every_n_frames,
        classify_every_n_frames=job.classify_every_n_frames,
        pops_classify_every_n_frames=job.pops_classify_every_n_frames,
        include_annotated_video=job.include_annotated_video,
        queued_at=job.queued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        result_url=result_url,
        annotated_video_url=annotated_video_url,
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_status(job)


@app.get("/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.SUCCEEDED or job.result_path is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status.value}; result not available.",
        )
    if not job.result_path.exists():
        raise HTTPException(status_code=410, detail="Result file has been GC'd")
    return FileResponse(
        job.result_path,
        media_type="application/json",
        filename="tracking.json",
    )


@app.get("/v1/jobs/{job_id}/annotated_video")
async def get_job_annotated_video(job_id: str):
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.SUCCEEDED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status.value}; annotated video not available.",
        )
    if not job.include_annotated_video or job.annotated_video_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "This job was not created with include_annotated_video=true, "
                "or the encode failed."
            ),
        )
    if not job.annotated_video_path.exists():
        raise HTTPException(status_code=410, detail="Annotated video has been GC'd")
    return FileResponse(
        job.annotated_video_path,
        media_type="video/mp4",
        filename="annotated.mp4",
    )


@app.delete("/v1/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    store = app.state.store
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        # Idempotent delete — remove artifacts and the record.
        shutil.rmtree(job.work_dir, ignore_errors=True)
        store.delete(job_id)
        return None
    if job.status == JobStatus.QUEUED:
        # Worker hasn't started yet. The status check at the top of
        # Worker._handle() will skip it once it pops off the queue, so we can
        # free the input video now instead of waiting for GC.
        store.update(job_id, status=JobStatus.CANCELLED)
        shutil.rmtree(job.work_dir, ignore_errors=True)
        store.delete(job_id)
        log.info("api.job_cancelled", job_id=job_id, prior_status="queued",
                 eager_cleanup=True)
        return None
    # Running: mark cancelled. We can't hard-stop the executor thread
    # (engine doesn't expose a cancellation hook), but the worker re-checks
    # status via update_if_not_cancelled before writing the terminal state
    # and will rmtree the work_dir itself.
    store.update(job_id, status=JobStatus.CANCELLED)
    log.info("api.job_cancelled", job_id=job_id, prior_status=job.status.value)
    return None
