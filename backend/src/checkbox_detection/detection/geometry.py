"""Pure box geometry helpers. Must not import anything outside `detection.types`."""

from checkbox_detection.detection.types import Box, Rect


def iou(a: Rect, b: Rect) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union else 0.0


def nms(rects: list[Rect], max_iou: float) -> list[Rect]:
    """Drop rects overlapping a kept one by more than `max_iou`; larger rects win."""
    kept: list[Rect] = []
    for r in sorted(rects, key=lambda r: (r.x2 - r.x1) * (r.y2 - r.y1), reverse=True):
        if all(iou(r, k) <= max_iou for k in kept):
            kept.append(r)
    return kept


def sort_reading_order(boxes: list[Box]) -> list[Box]:
    """Rows top to bottom, left to right inside a row.

    A box joins the current row when its vertical centre falls inside the row's first box, so a
    few px of skew does not swap boxes on one visual row.
    """
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.rect.y1 + b.rect.y2):
        first = rows[-1][0].rect if rows else None
        if first and first.y1 <= (b.rect.y1 + b.rect.y2) / 2 <= first.y2:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b.rect.x1)]
