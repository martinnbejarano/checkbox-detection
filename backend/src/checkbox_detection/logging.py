"""JSON log lines and the per-request middleware that writes one line per request."""

import json
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger("checkbox_detection")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Tags the response with `X-Request-ID` and logs status, timings and what the route recorded.

    Routes add their own fields (pages, decode_ms, ...) to `request.state.log_fields`.
    """
    request_id = uuid4().hex
    request.state.request_id = request_id
    start = perf_counter()
    error: Exception | None = None
    try:
        response = await call_next(request)
    except Exception as exc:
        error = exc
        # Built here, not in `api/errors.py`, so nothing below the wiring imports `api`.
        response = JSONResponse(
            {
                "type": "about:blank",
                "title": "Internal Server Error",
                "status": 500,
                "detail": f"Unexpected error; request id {request_id}",
            },
            status_code=500,
            media_type="application/problem+json",
        )
    response.headers["X-Request-ID"] = request_id
    fields = {
        "request_id": request_id,
        "status": response.status_code,
        "total_ms": round((perf_counter() - start) * 1000, 1),
        **getattr(request.state, "log_fields", {}),
    }
    logger.log(
        logging.ERROR if error else logging.INFO,
        "request",
        exc_info=error,
        extra={"fields": fields},
    )
    return response
