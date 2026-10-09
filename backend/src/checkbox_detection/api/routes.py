"""POST /detect and GET /health. The only module with FastAPI routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, UploadFile

from checkbox_detection.api.schemas import BoxOut, DetectResponse, HealthResponse
from checkbox_detection.detection import Detector
from checkbox_detection.imaging import load_pages

router = APIRouter()


def get_detector(request: Request) -> Detector:
    detector: Detector = request.app.state.detector
    return detector


# Plain `def`: FastAPI runs it in its threadpool, and OpenCV releases the GIL.
@router.post("/detect")
def detect(
    file: UploadFile, detector: Annotated[Detector, Depends(get_detector)]
) -> DetectResponse:
    pages = load_pages(file.file.read())
    return DetectResponse(
        boxes=[
            BoxOut.from_box(box, page.index)
            for page in pages
            for box in detector.detect(page.image)
        ]
    )


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")
