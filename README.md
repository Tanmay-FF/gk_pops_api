# gk_pops_api

Minimal async HTTP backend that takes a video and returns the `_tracking.json`
artifact.

## Architecture

`pops_api_v1/` ships with its own self-contained, headless tracking engine at
`pops_api_v1/api_engine/`. `HeadlessTrackingEngine.process_video()` returns a
single dict (the `_tracking.json` structure).

## API

Base path: `/v1`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/jobs` | Multipart upload (`video`, `camera_placement`, `json_every_n_frames`, optional `include_annotated_video`). Returns `{ job_id, status: "queued" }`. |
| `GET` | `/v1/jobs/{job_id}` | Status: `queued` / `running` / `succeeded` / `failed` / `cancelled`. Includes `result_url` and (if requested) `annotated_video_url` once succeeded. |
| `GET` | `/v1/jobs/{job_id}/result` | Streams the `tracking.json` (only when `status == "succeeded"`). |
| `GET` | `/v1/jobs/{job_id}/annotated_video` | Streams the annotated MP4 (only when the job was created with `include_annotated_video=true` and finished successfully). |
| `DELETE` | `/v1/jobs/{job_id}` | Cancels (or deletes if already terminal). |
| `GET` | `/v1/healthz` | Liveness. |
| `GET` | `/v1/readyz` | Readiness — 200 once the engine is constructed. |

Allowed `camera_placement` values:
- `outside_facing_entrance` (default)
- `inside_facing_exit`
- `inside_exit_on_right`
- `inside_exit_on_left`
- `inside_exit_on_both`

Classification runs every frame (`classify_every_n_frames` is fixed at `1`
inside the API and not exposed as a client parameter).

`include_annotated_video` is `false` by default. Set it to `true` on the
`POST /v1/jobs` form to also produce a Gradio-style annotated MP4 (bboxes,
trails, fill/bag/POPS chips, link lines, HUD) — download it from
`/v1/jobs/{job_id}/annotated_video` once the job succeeds. The JSON is
produced either way.

## Run locally (PowerShell)

```powershell
# from inside pops_api_v1/
cd pops_api_v1
$env:POPS_API_LOG_LEVEL = "DEBUG"
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

Install deps first with `pip install -r requirements.txt` (still from inside
`pops_api_v1/`).

Submit a job:

```powershell
$resp = curl.exe -s -F "video=@sample.mp4" `
    -F "camera_placement=outside_facing_entrance" `
    -F "json_every_n_frames=1" `
    -F "include_annotated_video=true" `
    http://localhost:8080/v1/jobs | ConvertFrom-Json
$jobId = $resp.job_id

# Poll until done
do {
    Start-Sleep -Seconds 2
    $status = curl.exe -s "http://localhost:8080/v1/jobs/$jobId" | ConvertFrom-Json
    Write-Host "status=$($status.status)"
} until ($status.status -in @('succeeded','failed','cancelled'))

# Download the JSON
curl.exe -s "http://localhost:8080/v1/jobs/$jobId/result" -o tracking.json

# Download the annotated MP4 (only if you asked for it on submit)
if ($status.annotated_video_url) {
    curl.exe -s "http://localhost:8080/v1/jobs/$jobId/annotated_video" -o annotated.mp4
}
```

The downloaded `tracking.json` follows the same schema as the existing
`<video>_tracking.json` artifact (top-level keys: `video_info`, `frames`,
`events`, `cart_classifications`, `pops_summary`, `summary`,
`processing_info`).

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `POPS_API_PORT` | `8080` | Port the server listens on. |
| `POPS_API_LOG_LEVEL` | `INFO` | uvicorn + structlog level. |
| `POPS_API_MAX_UPLOAD_MB` | `500` | Reject larger uploads (413). |
| `POPS_API_MAX_QUEUE` | `4` | Queue cap before 429. |
| `POPS_API_JOB_TTL_MIN` | `60` | Terminal jobs and their files are GC'd after this many minutes. |
| `POPS_API_CORS_ALLOW_ORIGINS` | `*` | Comma-separated allowlist for prod. |

## Tradeoffs documented as out-of-scope

These are deliberate v1 deferrals.

- No authentication.
- No multi-instance horizontal scaling (single-instance + in-memory store).
- No fine-grained progress reporting (status is binary `running`/`succeeded`).
- Uploads capped at `POPS_API_MAX_UPLOAD_MB` (default 500 MB).
