"""FillRatioClassifier: a box is checked when its interior holds enough mark ink."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from checkbox_detection.detection.binarize import (
    binarize,
    border_thickness,
    remove_lines,
    to_clean_gray,
)
from checkbox_detection.detection.types import Rect


@dataclass(frozen=True, slots=True)
class FillParams:
    # Tuned on validation by training/search.py fill.
    # Smallest inset, as a share of the side.
    min_margin_frac: float = 0.2
    # Gap kept past the measured border, as a share of the side.
    border_gap_frac: float = 0.05
    # Rules longer than this multiple of the side are removed around the box first.
    rule_length_factor: float = 1.5
    # A stroke counts as a mark only when at least this share of its ink is inside the box;
    # a stray stroke that only crosses the box is mostly outside.
    min_inside_share: float = 0.5
    # Straight runs at least this share of the side are box edges, set apart from the marks.
    edge_length_frac: float = 0.8
    # Interior ink ratio above this means checked: 0.05 plus a 0.02 margin, so an ambiguous ratio
    # resolves to unchecked. Kept although validation prefers 0.02, where a 2% speck
    # would read as checked.
    fill_threshold: float = 0.07


class FillRatioClassifier:
    def __init__(self) -> None:
        self.params = FillParams()

    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]:
        binary = binarize(to_clean_gray(image))
        ratios = [mark_ratio(binary, rect, self.params) for rect in rects]
        return [(ratio > self.params.fill_threshold, ratio) for ratio in ratios]


def mark_ratio(binary: np.ndarray, rect: Rect, params: FillParams) -> float:
    """Share of the interior covered by strokes that sit mostly inside the box."""
    w, h = rect.x2 - rect.x1, rect.y2 - rect.y1
    side = min(w, h)
    border = border_thickness(binary[rect.y1 : rect.y2, rect.x1 : rect.x2])
    margin = math.ceil(max(border + params.border_gap_frac * side, params.min_margin_frac * side))

    # A crop one side wider on every edge, without table rules, so a stroke can be followed out.
    cx1, cy1 = max(0, rect.x1 - side), max(0, rect.y1 - side)
    crop = binary[cy1 : rect.y2 + side, cx1 : rect.x2 + side]
    crop = remove_lines(crop, int(params.rule_length_factor * side))
    x1, y1, x2, y2 = rect.x1 - cx1, rect.y1 - cy1, rect.x2 - cx1, rect.y2 - cy1

    # Strokes without the box's own edges, so a mark and the border are separate components.
    strokes = remove_lines(crop, int(params.edge_length_frac * side))
    # Each stroke takes in the edge ink within about one border of it, which fills the gap where
    # it crossed the edge. A closing does not: at a shallow angle the gap along the stroke is
    # longer than the kernel, and the erosion cuts the bridge. +3 px keeps 1 px borders bridged.
    k = 2 * border + 3
    bridged = cv2.dilate(strokes, np.ones((k, k), np.uint8)) & crop
    n, labels = cv2.connectedComponents(bridged)

    total = np.bincount(labels[strokes > 0], minlength=n)
    box = (slice(y1, y2), slice(x1, x2))
    in_box = np.bincount(labels[box][strokes[box] > 0], minlength=n)
    is_mark = in_box >= params.min_inside_share * total
    is_mark[0] = True  # ink removed as a straight run inside the interior, e.g. part of a fill
    interior = (slice(y1 + margin, y2 - margin), slice(x1 + margin, x2 - margin))
    ink = (crop[interior] > 0) & is_mark[labels[interior]]
    return float(np.mean(ink)) if ink.size else 0.0
