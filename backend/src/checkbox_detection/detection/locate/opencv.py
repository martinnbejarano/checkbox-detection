"""OpenCVLocator: binarize, drop table rules, keep hollow near-square contours. Needs no weights."""

from dataclasses import dataclass

import cv2
import numpy as np

from checkbox_detection.detection.binarize import (
    binarize,
    border_thickness,
    remove_lines,
    to_clean_gray,
)
from checkbox_detection.detection.geometry import nms
from checkbox_detection.detection.types import Rect


@dataclass(frozen=True, slots=True)
class OpenCVParams:
    # Shape thresholds tuned on validation by training/search.py opencv. The size extremes and
    # the upscale rule are scale rules, not tuned.
    # Below this side a shape is noise and its state cannot be read.
    min_side_px: int = 8
    # Above this fraction of the page's short edge a shape is a table cell, not a checkbox.
    max_side_frac: float = 0.06
    # Checkboxes are square-ish; real forms run about 0.8-1.0 height/width.
    min_aspect: float = 0.7
    max_aspect: float = 1.4
    # approxPolyDP tolerance as a fraction of the contour perimeter.
    poly_epsilon_frac: float = 0.05
    # Contour area / bbox area; a rectangle is close to 1, glyphs like "O" are lower.
    min_rectangularity: float = 0.85
    # With 4 vertices and at least this rectangularity a candidate is confident; glyphs such as
    # "D" or "0" reach about 0.84 and stay weak.
    confident_rectangularity: float = 0.86
    # A confident candidate also has a thin border: at most this share of its side.
    # 0.15 is the top of the range tried; try higher when the validation data grows.
    max_confident_border_frac: float = 0.15
    # Hollow: the contour's holes cover at least this share of its bbox. An X or a fill with a
    # gap leaves holes; a solid blob or a thick glyph does not.
    min_hole_frac: float = 0.35
    # Rules are erased when longer than this multiple of the largest allowed box side, so box
    # edges survive.
    line_length_factor: float = 1.5
    # Weak candidates (not confident) are kept only within this band around the median side of
    # the confident ones.
    median_band: tuple[float, float] = (0.85, 1.2)
    # Fewer confident candidates than this cannot set the median, and weak ones are dropped.
    min_confident: int = 5
    # Pages whose short edge is below this are upscaled x2 first (low-resolution photos, crops).
    upscale_below_px: int = 1000
    # Duplicates (a box and its inner border) overlap more than this.
    nms_iou: float = 0.3


@dataclass(frozen=True, slots=True)
class _Candidate:
    rect: Rect
    confident: bool


class OpenCVLocator:
    def __init__(self) -> None:
        self.params = OpenCVParams()

    def locate(self, image: np.ndarray) -> list[Rect]:
        scale = 2 if min(image.shape[:2]) < self.params.upscale_below_px else 1
        if scale > 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        binary = binarize(to_clean_gray(image))
        max_side = self.params.max_side_frac * min(binary.shape)
        binary = remove_lines(binary, int(self.params.line_length_factor * max_side))
        candidates = find_candidates(binary, self.params)
        rects = nms(filter_by_page_size(candidates, self.params), self.params.nms_iou)
        return [
            Rect(r.x1 // scale, r.y1 // scale, r.x2 // scale, r.y2 // scale, 1.0) for r in rects
        ]


def find_candidates(binary: np.ndarray, params: OpenCVParams) -> list[_Candidate]:
    """Outer contours that are near-square, rectangular and hollow."""
    max_side = params.max_side_frac * min(binary.shape)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    hole_area: dict[int, float] = {}
    for i, (_, _, _, parent) in enumerate(hierarchy[0]):
        if parent != -1:
            hole_area[parent] = hole_area.get(parent, 0.0) + cv2.contourArea(contours[i])

    found = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  # a hole, not an outer boundary
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (params.min_side_px <= min(w, h) and max(w, h) <= max_side):
            continue
        if not params.min_aspect <= h / w <= params.max_aspect:
            continue
        if hole_area.get(i, 0.0) < params.min_hole_frac * w * h:
            continue
        perimeter = cv2.arcLength(contour, True)
        four_sided = len(cv2.approxPolyDP(contour, params.poly_epsilon_frac * perimeter, True)) == 4
        # contourArea runs through pixel centres, so a perfect w x h box measures (w-1)(h-1).
        rectangularity = cv2.contourArea(contour) / ((w - 1) * (h - 1))
        if four_sided or rectangularity >= params.min_rectangularity:
            thin = border_thickness(binary[y : y + h, x : x + w]) <= (
                params.max_confident_border_frac * min(w, h)
            )
            confident = four_sided and rectangularity >= params.confident_rectangularity and thin
            found.append(_Candidate(Rect(x, y, x + w, y + h, 1.0), confident))
    return found


def filter_by_page_size(candidates: list[_Candidate], params: OpenCVParams) -> list[Rect]:
    """Keep confident candidates; keep weak ones only near the page's own median box side."""
    sides = [max(c.rect.x2 - c.rect.x1, c.rect.y2 - c.rect.y1) for c in candidates if c.confident]
    if len(sides) < params.min_confident:
        return [c.rect for c in candidates if c.confident]
    low, high = (f * float(np.median(sides)) for f in params.median_band)
    return [
        c.rect
        for c in candidates
        if c.confident or low <= max(c.rect.x2 - c.rect.x1, c.rect.y2 - c.rect.y1) <= high
    ]
