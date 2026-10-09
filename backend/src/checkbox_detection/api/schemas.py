"""Response models for the HTTP contract (spec section 2)."""

from pydantic import BaseModel

from checkbox_detection.detection.types import Box


class BoxOut(BaseModel):
    bbox: tuple[int, int, int, int]
    is_checked: bool
    page: int

    @classmethod
    def from_box(cls, box: Box, page: int) -> "BoxOut":
        return cls(bbox=(box.x1, box.y1, box.x2, box.y2), is_checked=box.is_checked, page=page)


class DetectResponse(BaseModel):
    boxes: list[BoxOut]


class HealthResponse(BaseModel):
    status: str
