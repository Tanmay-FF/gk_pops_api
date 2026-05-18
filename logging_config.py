"""Structured JSON logging — one line per event, request-id and job-id contextual.

Cloud Run automatically ships stdout to Cloud Logging; the JSON format means
log fields become queryable structured payloads with no extra wiring.
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_job_id_ctx: ContextVar[str | None] = ContextVar("job_id", default=None)


def bind_job_id(job_id: str | None) -> None:
    _job_id_ctx.set(job_id)


def _inject_ctx(_logger, _name, event_dict):
    rid = _request_id_ctx.get()
    jid = _job_id_ctx.get()
    if rid:
        event_dict["request_id"] = rid
    if jid:
        event_dict["job_id"] = jid
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _inject_ctx,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Tag every incoming request with a UUID4, surface it as X-Request-ID."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = _request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            _request_id_ctx.reset(token)
