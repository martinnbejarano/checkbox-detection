"""Response models for the HTTP contract."""

from pydantic import BaseModel

from checkbox_detection.detection.types import Box


class BoxOut(BaseModel):
    bbox: tuple[int, int, int, int]
    is_checked: bool
    page: int
    score: float  # locator confidence in [0, 1]; OpenCV always 1.0
    label: str | None = None  # serialized only when set: see the `/detect` route

    @classmethod
    def from_box(cls, box: Box, page: int, labels: bool = False) -> "BoxOut":
        r = box.rect
        out = cls(
            bbox=(r.x1, r.y1, r.x2, r.y2),
            is_checked=box.is_checked,
            page=page,
            score=round(r.score, 3),
        )
        if labels:
            out.label = box.label
        return out


class DetectResponse(BaseModel):
    boxes: list[BoxOut]


class PageOut(BaseModel):
    """One line of `/detect/stream`: a page's boxes, sent as soon as that page is done."""

    page: int
    boxes: list[BoxOut]


class HealthResponse(BaseModel):
    status: str
