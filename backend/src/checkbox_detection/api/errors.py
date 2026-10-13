"""Maps imaging errors, HTTP errors and validation errors to RFC 9457 problem+json."""

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from checkbox_detection.imaging.errors import (
    CorruptInput,
    ImagingError,
    InputTooLarge,
    UnsupportedFormat,
)

_IMAGING_STATUS = {UnsupportedFormat: 415, InputTooLarge: 413, CorruptInput: 422}


def problem(status: int, detail: str) -> JSONResponse:
    body = {"type": "about:blank", "title": HTTPStatus(status).phrase, "status": status}
    return JSONResponse(
        {**body, "detail": detail}, status_code=status, media_type="application/problem+json"
    )


def _imaging_error(request: Request, exc: ImagingError) -> JSONResponse:
    return problem(_IMAGING_STATUS[type(exc)], str(exc))


def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return problem(exc.status_code, str(exc.detail))


def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
    )
    return problem(422, detail)


# Starlette types every handler's exception as `Exception`; `Any` lets each one take its own class.
_HANDLERS: dict[type[Exception], Callable[[Request, Any], JSONResponse]] = {
    ImagingError: _imaging_error,
    HTTPException: _http_error,
    RequestValidationError: _validation_error,
}


def add_error_handlers(app: FastAPI) -> None:
    for exc_type, handler in _HANDLERS.items():
        app.add_exception_handler(exc_type, handler)
