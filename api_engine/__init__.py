"""api_engine — headless, JSON-only fork of `engine/`.

This package is a self-contained copy of the POPS pipeline with every UI,
rendering, BEV, pose, VLM, and analytics dependency removed. It exposes a
single `HeadlessTrackingEngine.process_video()` that returns the same
_tracking.json structure produced by the original engine, but skips all
work that the JSON does not need.

Why a fork instead of importing engine/:
  - The original engine imports gradio at module level and always runs pose
    estimation; both are wasted work for our API. Forking lets us delete
    them without touching the Gradio demo or the older api/main.py service.
"""
from .tracker import HeadlessTrackingEngine

__all__ = ["HeadlessTrackingEngine"]
