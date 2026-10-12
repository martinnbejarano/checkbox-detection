"""POST /detect and GET /health. The only module with FastAPI routes."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from checkbox_detection.api.schemas import BoxOut, DetectResponse, HealthResponse
from checkbox_detection.config import Settings
from checkbox_detection.detection import Pipeline
from checkbox_detection.imaging import load_pages

router = APIRouter()


def get_pipeline(request: Request) -> Pipeline:
    pipeline: Pipeline = request.app.state.pipeline
    return pipeline


# Unset fields are left out, so `label` appears only when labels were asked for.
@router.post("/detect", response_model_exclude_unset=True)
def detect(
    request: Request,
    file: UploadFile,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    labels: bool = False,
) -> DetectResponse:

    settings: Settings = request.app.state.settings
    data = file.file.read()

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"Upload is above the {settings.max_upload_bytes} byte limit")

    start = perf_counter()
    pages = load_pages(data, settings)
    decoded = perf_counter()
    boxes: list[BoxOut] = []
    label_s = 0.0
    for page in pages:
        upright = pipeline.straighten(page.image)
        page_boxes = pipeline.run(upright.image)
        if labels:
            labeled = perf_counter()
            page_boxes = pipeline.label(upright.image, page_boxes)
            label_s += perf_counter() - labeled
        page_boxes = upright.boxes_to_original(page_boxes)
        boxes += [BoxOut.from_box(box, page.index, labels) for box in page_boxes]
    request.state.log_fields = {
        "pages": len(pages),
        "decode_ms": round((decoded - start) * 1000, 1),
        "detect_ms": round((perf_counter() - decoded - label_s) * 1000, 1),
        "label_ms": round(label_s * 1000, 1),
        "n_boxes": len(boxes),
    }
    return DetectResponse(boxes=boxes)


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")
