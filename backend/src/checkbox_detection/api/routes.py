"""POST /detect and GET /health. The only module with FastAPI routes."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from checkbox_detection.api.schemas import BoxOut, DetectResponse, HealthResponse
from checkbox_detection.config import Settings
from checkbox_detection.detection import Detector
from checkbox_detection.imaging import load_pages

router = APIRouter()


def get_detector(request: Request) -> Detector:
    detector: Detector = request.app.state.detector
    return detector


# Plain `def`: FastAPI runs it in its threadpool, and OpenCV releases the GIL.
@router.post("/detect")
def detect(
    request: Request, file: UploadFile, detector: Annotated[Detector, Depends(get_detector)]
) -> DetectResponse:
    settings: Settings = request.app.state.settings
    data = file.file.read()
    # Fallback: the body-limit middleware normally rejects oversized uploads before this.
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"Upload is above the {settings.max_upload_bytes} byte limit")

    start = perf_counter()
    pages = load_pages(data, settings)
    decoded = perf_counter()
    boxes = [
        BoxOut.from_box(box, page.index) for page in pages for box in detector.detect(page.image)
    ]
    request.state.log_fields = {
        "pages": len(pages),
        "decode_ms": round((decoded - start) * 1000, 1),
        "detect_ms": round((perf_counter() - decoded) * 1000, 1),
        "n_boxes": len(boxes),
    }
    return DetectResponse(boxes=boxes)


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")
