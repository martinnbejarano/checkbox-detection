"""OpenCVDetector: binarize, find hollow near-square contours, classify by interior ink."""

from dataclasses import dataclass

import cv2
import numpy as np

from checkbox_detection.detection.geometry import sort_reading_order
from checkbox_detection.detection.types import Box


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
    # Adaptive-threshold block as a fraction of page width, so it scales with resolution.
    block_frac: float = 0.02
    # A pixel is ink when it is this much darker than its neighbourhood mean.
    threshold_offset: int = 10
    # Interior is the box shrunk by this fraction of its side on every edge, skipping the border.
    interior_margin_frac: float = 0.2
    # A border ring must be mostly ink and clearly darker than the interior to count as hollow.
    min_ring_ink: float = 0.5
    # Interior ink ratio above this means checked.
    fill_threshold: float = 0.10


class OpenCVDetector:
    def __init__(self, params: OpenCVParams | None = None) -> None:
        self.params = params or OpenCVParams()

    def detect(self, image: np.ndarray) -> list[Box]:
        binary = binarize(to_gray(image), self.params)
        boxes = []
        for x, y, w, h in find_candidates(binary, self.params):
            ratio = fill_ratio(binary, x, y, w, h, self.params)
            boxes.append(Box(x, y, x + w, y + h, ratio > self.params.fill_threshold, ratio))
        return sort_reading_order(boxes)


def to_gray(image: np.ndarray) -> np.ndarray:
    # Dropping saturated overlays (red watermark, blue fill) arrives with real-form tuning (#18).
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def binarize(gray: np.ndarray, params: OpenCVParams) -> np.ndarray:
    """Ink is 255, paper is 0."""
    block = max(3, int(gray.shape[1] * params.block_frac) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        params.threshold_offset,
    )


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
    return ring_ink >= params.min_ring_ink and ring_ink > fill_ratio(binary, x, y, w, h, params)


def fill_ratio(binary: np.ndarray, x: int, y: int, w: int, h: int, params: OpenCVParams) -> float:
    """Share of ink pixels inside the box, away from its border."""
    mx, my = int(w * params.interior_margin_frac), int(h * params.interior_margin_frac)
    interior = binary[y + my : y + h - my, x + mx : x + w - mx]
    return float(np.mean(interior > 0)) if interior.size else 0.0
