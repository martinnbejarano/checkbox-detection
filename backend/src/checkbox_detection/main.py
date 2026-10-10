from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, Request, Response
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from checkbox_detection.api.errors import add_error_handlers, problem
from checkbox_detection.api.routes import router
from checkbox_detection.config import Settings
from checkbox_detection.detection import build_pipeline
from checkbox_detection.logging import configure_logging, log_requests


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cv2.setNumThreads(settings.cv_threads)
        app.state.pipeline = build_pipeline(
            settings.locator, settings.classifier, settings.models_dir, settings.onnx_threads
        )
        yield

    async def reject_declared_oversize(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # RequestBodyLimitMiddleware answers an oversized Content-Length in plain text,
        # so that case is answered here as problem+json first.
        if int(request.headers.get("content-length") or 0) > settings.max_upload_bytes:
            return problem(413, f"Upload is above the {settings.max_upload_bytes} byte limit")
        return await call_next(request)

    app = FastAPI(title="Checkbox detection", lifespan=lifespan)
    app.state.settings = settings
    # Last added runs first: log_requests wraps everything, then the Content-Length check,
    # then the byte counter that stops streamed bodies.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_upload_bytes)
    app.middleware("http")(reject_declared_oversize)
    app.middleware("http")(log_requests)
    add_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
