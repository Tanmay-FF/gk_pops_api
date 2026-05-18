"""pops_api — minimal async backend that turns a video into a POPS _tracking.json.

This package is isolated from the existing Gradio demo (app_poc_v2.py) and the
richer FastAPI service at api/main.py. It depends on engine.TrackingEngine but
does not modify the engine beyond two optional backward-compatible kwargs on
process_video.
"""
__version__ = "0.1.0"
