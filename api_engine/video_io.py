"""Video I/O — read frames; optionally write an annotated AVI and re-encode it
to a browser-friendly MP4 when the caller asks for it.

The writer/reencoder are only used when the API caller opts into the
annotated MP4 — for JSON-only jobs we skip them entirely so the pipeline
stays light.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2


def open_video(path: str):
    """Open a video file. Returns (cap, width, height, fps, total_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, fps, total


def create_writer(avi_path: Path, w: int, h: int, fps: int):
    """Create an XVID AVI writer at the given path. Returns the writer."""
    avi_path = Path(avi_path)
    avi_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(avi_path), cv2.VideoWriter_fourcc(*"XVID"), int(fps), (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {avi_path}")
    return writer


# Detect NVENC support lazily — probing ffmpeg on every job adds ~50ms startup
# per request, but caching it module-wide is fine because the encoder set
# doesn't change after process start.
_NVENC_AVAILABLE: Optional[bool] = None


def _ffmpeg_exe() -> str:
    """Resolve an ffmpeg binary. Prefer imageio_ffmpeg's bundled build (it's
    pinned to a known-good NVENC-capable version) and fall back to the system
    PATH so a slim base image without imageio_ffmpeg still works."""
    try:
        import imageio_ffmpeg  # imported lazily — only needed when writing video.
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _check_nvenc() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        r = subprocess.run(
            [_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in (r.stdout or "")
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def reencode_to_mp4(avi_path: Path, mp4_path: Path,
                    delete_source: bool = True) -> Path:
    """Re-encode an AVI to H.264 MP4 (NVENC when available, libx264 fallback).

    Removes the AVI on success unless delete_source=False. Raises on failure.
    """
    avi_path = Path(avi_path)
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if mp4_path.exists():
        mp4_path.unlink()

    ffmpeg = _ffmpeg_exe()
    if _check_nvenc():
        cmd = [
            ffmpeg, "-y", "-i", str(avi_path),
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
            "-cq", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4_path),
        ]
    else:
        cmd = [
            ffmpeg, "-y", "-i", str(avi_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4_path),
        ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        # Keep the AVI on failure so the job artifacts can still be inspected.
        raise RuntimeError(
            f"ffmpeg re-encode failed (rc={proc.returncode}): "
            f"{(proc.stderr or b'').decode(errors='replace')[-500:]}"
        )
    if delete_source and avi_path.exists():
        try:
            os.remove(avi_path)
        except OSError:
            pass
    return mp4_path
