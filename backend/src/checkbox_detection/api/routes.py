"""POST /detect, POST /detect/stream and GET /health. The only module with FastAPI routes."""

import json
from collections.abc import Iterator
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import StreamingResponse

from checkbox_detection.api.schemas import BoxOut, DetectResponse, HealthResponse, PageOut
from checkbox_detection.config import Settings
from checkbox_detection.detection import Pipeline
from checkbox_detection.imaging import Page, iter_pages
from checkbox_detection.imaging.errors import ImagingError
from checkbox_detection.logging import logger

router = APIRouter()


def get_pipeline(request: Request) -> Pipeline:
    pipeline: Pipeline = request.app.state.pipeline
    return pipeline


def _detect_pages(
    pages: Iterator[Page], pipeline: Pipeline, labels: bool, fields: dict[str, float]
) -> Iterator[list[BoxOut]]:
    """Each page's boxes as soon as that page is done, adding its timings to `fields`."""
    while True:
        start = perf_counter()
        page = next(pages, None)
        if page is None:
            return
        decoded = perf_counter()
        boxes, label_s = pipeline.detect(page.image, labels)
        fields["decode_ms"] += (decoded - start) * 1000
        fields["detect_ms"] += (perf_counter() - decoded - label_s) * 1000
        fields["label_ms"] += label_s * 1000
        fields["n_boxes"] += len(boxes)
        yield [BoxOut.from_box(box, page.index, labels) for box in boxes]


def _open(request: Request, file: UploadFile) -> tuple[Iterator[Page], dict[str, float]]:
    """The upload's pages, checked (imaging errors raise here, as problem+json), and the log
    fields `_detect_pages` fills in."""
    settings: Settings = request.app.state.settings
    data = file.file.read()
    start = perf_counter()
    n_pages, pages = iter_pages(data, settings)
    fields = {
        "pages": n_pages,
        "decode_ms": (perf_counter() - start) * 1000,
        "detect_ms": 0.0,
        "label_ms": 0.0,
        "n_boxes": 0,
    }
    return pages, fields


def _rounded(fields: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 1) for key, value in fields.items()}


# Unset fields are left out, so `label` appears only when labels were asked for.
@router.post("/detect", response_model_exclude_unset=True)
def detect(
    request: Request,
    file: UploadFile,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    labels: bool = False,
) -> DetectResponse:
    pages, fields = _open(request, file)
    boxes = [box for page in _detect_pages(pages, pipeline, labels, fields) for box in page]
    request.state.log_fields = _rounded(fields)
    return DetectResponse(boxes=boxes)


@router.post("/detect/stream", response_class=StreamingResponse)
def detect_stream(
    request: Request,
    file: UploadFile,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    labels: bool = False,
) -> StreamingResponse:
    """The same boxes as `/detect`, as NDJSON: one `{"page", "boxes"}` line per page, in page
    order, each sent as soon as that page is done. An upload that fails its checks gets the same
    problem+json error as `/detect`; a failure after the first line ends the stream with one
    `{"error": "..."}` line."""
    pages, fields = _open(request, file)
    request.state.log_fields = {"pages": fields["pages"]}
    request_id: str = request.state.request_id

    def lines() -> Iterator[str]:
        try:
            for index, boxes in enumerate(_detect_pages(pages, pipeline, labels, fields)):
                yield PageOut(page=index, boxes=boxes).model_dump_json(exclude_unset=True) + "\n"
        except ImagingError as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
        except Exception:
            logger.exception("stream failed", extra={"fields": {"request_id": request_id}})
            yield json.dumps({"error": f"Unexpected error; request id {request_id}"}) + "\n"
        # The request line was written when the stream started; this one has the timings.
        logger.info("stream", extra={"fields": {"request_id": request_id, **_rounded(fields)}})

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")
