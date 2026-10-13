"""Checkbox detection on one decoded page: pure NumPy/OpenCV, no I/O. Read this file first.

`Pipeline.detect` runs a page through every stage, in this order:
  0. straighten - the page upright and level, boxes mapped back after (`orient.py`, `deskew.py`)
  1. locate   - `Locator`: where are the boxes          (`locate/`)
  2. classify - `StateClassifier`: is each one checked (`classify/`)
  3. label    - `RapidOcrLabeler`: what does it say     (`label.py`, only when labels are asked for)

`detection` imports nothing from `api`, `imaging` or `config`, and the stages import nothing from
each other: `binarize.py`, `geometry.py` and `types.py` hold what they share.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from checkbox_detection.detection.binarize import to_clean_gray
from checkbox_detection.detection.classify import StateClassifier
from checkbox_detection.detection.classify.fill_ratio import FillRatioClassifier
from checkbox_detection.detection.classify.state_cnn import StateCnnClassifier
from checkbox_detection.detection.deskew import Upright, deskew
from checkbox_detection.detection.geometry import sort_reading_order
from checkbox_detection.detection.label import RapidOcrLabeler
from checkbox_detection.detection.locate import Locator
from checkbox_detection.detection.locate.opencv import OpenCVLocator
from checkbox_detection.detection.locate.yolox import YoloxLocator
from checkbox_detection.detection.orient import (
    half_turn,
    is_sideways,
    is_upside_down,
    quarter_turn,
    text_lines,
)
from checkbox_detection.detection.types import Box


@dataclass(frozen=True)
class Pipeline:
    locator: Locator
    classifier: StateClassifier
    labeler: RapidOcrLabeler

    def detect(self, image: np.ndarray, labels: bool = False) -> tuple[list[Box], float]:
        """The boxes on `image`, in its pixels and reading order, labelled when `labels`; and the
        seconds stage 3 took, so callers can time it apart."""
        upright = self.straighten(image)
        # Clean gray once for stages 1-3, which take it as is. After stage 0, not before: deskew
        # interpolates the page, and interpolated colour edges clean differently.
        page = to_clean_gray(upright.image)
        rects = self.locator.locate(page)
        states = self.classifier.classify(page, rects)
        boxes = sort_reading_order([Box(r, c) for r, (c, _) in zip(rects, states, strict=True)])
        label_s = 0.0
        if labels:
            start = perf_counter()
            texts = self.labeler.label(page, [b.rect for b in boxes])
            boxes = [replace(b, label=t) for b, t in zip(boxes, texts, strict=True)]
            label_s = perf_counter() - start
        return upright.boxes_to_original(boxes), label_s

    def straighten(self, image: np.ndarray) -> Upright:
        """Stage 0: the stages take `upright.image`; `upright.boxes_to_original` maps their boxes
        back to `image`. Quarter turn, then deskew, then half turn: upside down is told apart on a
        level page only."""
        ocr = self.labeler.ocr
        lines = text_lines(image, ocr)
        upright = quarter_turn(image) if is_sideways(lines) else Upright.identity(image)
        upright = upright.then(deskew(upright.image))
        if is_upside_down(upright, lines, ocr):
            upright = upright.then(half_turn(upright.image))
        return upright


LOCATORS: dict[str, Callable[[Path, int], Locator]] = {
    "opencv": lambda models_dir, threads: OpenCVLocator(),
    "yolox": YoloxLocator,
}
CLASSIFIERS: dict[str, Callable[[Path, int], StateClassifier]] = {
    "fill_ratio": lambda models_dir, threads: FillRatioClassifier(),
    "cnn": StateCnnClassifier,
}


def build_pipeline(locator: str, classifier: str, models_dir: Path, threads: int) -> Pipeline:
    if locator not in LOCATORS:
        raise ValueError(f"Unknown locator {locator!r}; choose one of {sorted(LOCATORS)}")
    if classifier not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier {classifier!r}; choose one of {sorted(CLASSIFIERS)}")
    return Pipeline(
        LOCATORS[locator](models_dir, threads),
        CLASSIFIERS[classifier](models_dir, threads),
        RapidOcrLabeler(threads),
    )
