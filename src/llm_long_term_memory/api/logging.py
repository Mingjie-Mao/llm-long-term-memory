"""One structured JSON event per request.

The fields are chosen so that a production question can be answered from the log
alone, without reproducing the request:

* *why was this slow?* — `latency_ms`, split into retrieval and total
* *what did it cost?* — `llm_calls`, `input_tokens`, `output_tokens`
* *which build produced this?* — `config` fingerprint
* *did it return anything?* — `retrieved`, `rejected`

Cost in currency is deliberately absent. This project runs on a free tier with no
published price schedule, and a USD figure invented from a list price would be a
number that looks authoritative and is not (D-cost, docs/DECISIONS.md).

Nothing here logs memory content or a credential. A memory body is user data; an
id and a count are enough to debug with.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar

import structlog

# Set per request, so nested log calls inherit it without threading an argument
# through the service layer.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """JSON to stdout by default; human-readable when a TTY asks for it."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def logger():
    return structlog.get_logger("llm_long_term_memory.api")


# Set once at startup. A module-level slot rather than a middleware constructor
# argument because the middleware stack is built before the service exists, so a
# value passed at `add_middleware` time is necessarily empty.
_fingerprint = ""


def set_fingerprint(value: str) -> None:
    global _fingerprint
    _fingerprint = value


class RequestLoggingMiddleware:
    """Pure ASGI middleware.

    Written at the ASGI level rather than as a BaseHTTPMiddleware subclass because
    the latter buffers the response body to measure it, which would make the latency
    this is trying to report partly its own.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger().info(
                "request",
                request_id=request_id,
                method=scope.get("method"),
                route=scope.get("path"),
                status=status_code,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                config=_fingerprint,
            )
            request_id_var.reset(token)
