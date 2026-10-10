"""Maps imaging errors, HTTP errors and validation errors to RFC 9457 problem+json."""

from http import HTTPStatus

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


def _imaging_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ImagingError)
    return problem(_IMAGING_STATUS[type(exc)], str(exc))


def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    return problem(exc.status_code, str(exc.detail))


def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    detail = "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
    )
    return problem(422, detail)


def add_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ImagingError, _imaging_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
