"""OpenCVLocator: binarize, then keep hollow near-square contours. The baseline; needs no weights.

Must not import `classify`, `label` or anything outside `detection`.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from checkbox_detection.detection.binarize import binarize, interior_ink, to_clean_gray
from checkbox_detection.detection.types import Rect


@dataclass(frozen=True, slots=True)
class OpenCVParams:
    # Below this side a shape is noise and its state cannot be read.
    min_side_px: int = 8
    # Above this fraction of the page's short edge a shape is a table cell, not a checkbox.
    max_side_frac: float = 0.06
    # Checkboxes are square-ish; real forms run about 0.8-1.0 height/width.
    min_aspect: float = 0.7
    max_aspect: float = 1.4
    # Contour area / bbox area; a rectangle is close to 1, glyphs like "O" are lower.
    min_rectangularity: float = 0.8
    # Interior used by the hollowness check: the box shrunk by this fraction of its side.
    interior_margin_frac: float = 0.2
    # A border ring must be mostly ink and clearly darker than the interior to count as hollow.
    min_ring_ink: float = 0.5


class OpenCVLocator:
    def __init__(self, params: OpenCVParams | None = None) -> None:
        self.params = params or OpenCVParams()

    def locate(self, image: np.ndarray) -> list[Rect]:
        binary = binarize(to_clean_gray(image))
        return [
            Rect(x, y, x + w, y + h, 1.0) for x, y, w, h in find_candidates(binary, self.params)
        ]


def find_candidates(binary: np.ndarray, params: OpenCVParams) -> list[tuple[int, int, int, int]]:
    """Outer contours that are near-square, rectangular and hollow, as (x, y, w, h)."""
    max_side = params.max_side_frac * min(binary.shape)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0] if contours else [], strict=True):
        if parent != -1:  # a hole, not an outer boundary
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (params.min_side_px <= min(w, h) and max(w, h) <= max_side):
            continue
        if not params.min_aspect <= h / w <= params.max_aspect:
            continue
        if cv2.contourArea(contour) / (w * h) < params.min_rectangularity:
            continue
        if not is_hollow(binary, x, y, w, h, params):
            continue
        rects.append((x, y, w, h))
    return rects


def is_hollow(binary: np.ndarray, x: int, y: int, w: int, h: int, params: OpenCVParams) -> bool:
    edges = np.concatenate(
        [
            binary[y, x : x + w],
            binary[y + h - 1, x : x + w],
            binary[y : y + h, x],
            binary[y : y + h, x + w - 1],
        ]
    )
    ring_ink = float(np.mean(edges > 0))
    interior = interior_ink(binary, x, y, w, h, params.interior_margin_frac)
    return ring_ink >= params.min_ring_ink and ring_ink > interior
