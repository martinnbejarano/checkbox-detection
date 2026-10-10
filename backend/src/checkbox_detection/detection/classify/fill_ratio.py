"""FillRatioClassifier: a box is checked when its interior holds enough ink.

Must not import `locate`, `label` or anything outside `detection`.
"""

from dataclasses import dataclass

import numpy as np

from checkbox_detection.detection.binarize import binarize, interior_ink, to_clean_gray
from checkbox_detection.detection.types import Rect


@dataclass(frozen=True, slots=True)
class FillParams:
    # Interior is the box shrunk by this fraction of its side on every edge, skipping the border.
    interior_margin_frac: float = 0.2
    # Interior ink ratio above this means checked.
    fill_threshold: float = 0.10


class FillRatioClassifier:
    def __init__(self, params: FillParams | None = None) -> None:
        self.params = params or FillParams()

    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]:
        binary = binarize(to_clean_gray(image))
        states = []
        for r in rects:
            ratio = interior_ink(
                binary, r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1, self.params.interior_margin_frac
            )
            states.append((ratio > self.params.fill_threshold, ratio))
        return states
