"""Checkbox detection on one decoded page: pure NumPy/OpenCV, no I/O. Read this file first.

A page is first turned upright (`straighten`), then goes through three stages, in this order:
  0. straighten - the page upright and level, boxes mapped back after (`orient.py`, `deskew.py`)
  1. locate   - `Locator`: where are the boxes          (`locate/`)
  2. classify - `StateClassifier`: is each one checked (`classify/`)
  3. label    - `RapidOcrLabeler`: what does it say     (`label.py`, only when labels are asked for)

Must not import FastAPI, `api`, `imaging` or `config`; only its own modules.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

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

    def straighten(self, image: np.ndarray) -> Upright:
        """Stage 0: `run` and `label` take `upright.image`; `upright.boxes_to_original` maps
        their boxes back to `image`. Quarter turn, then deskew, then half turn: upside down is
        told apart on a level page only."""
        ocr = self.labeler.ocr
        lines = text_lines(image, ocr)
        upright = quarter_turn(image) if is_sideways(lines) else Upright.identity(image)
        upright = upright.then(deskew(upright.image))
        if is_upside_down(upright, lines, ocr):
            upright = upright.then(half_turn(upright.image))
        return upright

    def run(self, image: np.ndarray) -> list[Box]:
        rects = self.locator.locate(image)
        states = self.classifier.classify(image, rects)
        return sort_reading_order([Box(r, c, s) for r, (c, s) in zip(rects, states, strict=True)])

    def label(self, image: np.ndarray, boxes: list[Box]) -> list[Box]:
        """Stage 3, separate from `run` so callers can time it and skip it when not asked for."""
        texts = self.labeler.label(image, [b.rect for b in boxes])
        return [replace(b, label=t) for b, t in zip(boxes, texts, strict=True)]


# Each constructor takes (models_dir, onnx_threads); the ONNX stages use them.
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
