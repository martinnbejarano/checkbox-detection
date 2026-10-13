"""Pure box geometry helpers."""

import math

import numpy as np

from checkbox_detection.detection.types import Box, Rect


def iou(a: Rect, b: Rect) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union else 0.0


def match(truth: list[Rect], found: list[Rect], min_iou: float) -> list[tuple[int, int]]:
    """Greedy one-to-one (truth index, found index) pairs, highest IoU first, IoU >= min_iou."""
    candidates = sorted(
        ((iou(t, f), ti, fi) for ti, t in enumerate(truth) for fi, f in enumerate(found)),
        reverse=True,
    )
    pairs: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_f: set[int] = set()
    for score, ti, fi in candidates:
        if score < min_iou:
            break
        if ti not in used_t and fi not in used_f:
            pairs.append((ti, fi))
            used_t.add(ti)
            used_f.add(fi)
    return sorted(pairs)


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


def map_rect(rect: Rect, matrix: np.ndarray, width: int, height: int) -> Rect:
    """The axis-aligned box around the 4 corners of `rect` mapped by a 2x3 affine `matrix`,
    clamped to a `width` x `height` page."""
    xs, ys = (rect.x1, rect.x2), (rect.y1, rect.y2)
    corners = np.array([[x, y, 1] for x in xs for y in ys]) @ matrix.T
    (x1, y1), (x2, y2) = corners.min(axis=0), corners.max(axis=0)
    return Rect(
        min(max(round(x1), 0), width),
        min(max(round(y1), 0), height),
        min(max(round(x2), 0), width),
        min(max(round(y2), 0), height),
        rect.score,
    )


def tile_windows(
    width: int, height: int, tile: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Windows (x1, y1, x2, y2) of at most `tile` px that cover the page.

    Neighbours overlap by at least `overlap` px, so a box smaller than that is whole in one window.
    """

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        n = math.ceil((length - tile) / (tile - overlap)) + 1
        return [round(i * (length - tile) / (n - 1)) for i in range(n)]

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]
