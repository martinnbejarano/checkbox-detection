from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI

from checkbox_detection.api.routes import router
from checkbox_detection.config import Settings
from checkbox_detection.detection import build_detector


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cv2.setNumThreads(settings.cv_threads)
        app.state.detector = build_detector(settings.detector)
        yield

    app = FastAPI(title="Checkbox detection", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
