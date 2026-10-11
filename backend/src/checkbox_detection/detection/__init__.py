"""Checkbox detection on one decoded page: pure NumPy/OpenCV, no I/O. Read this file first.

A page goes through three stages, in this order:
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
from checkbox_detection.detection.geometry import sort_reading_order
from checkbox_detection.detection.label import RapidOcrLabeler
from checkbox_detection.detection.locate import Locator
from checkbox_detection.detection.locate.opencv import OpenCVLocator
from checkbox_detection.detection.locate.yolox import YoloxLocator
from checkbox_detection.detection.types import Box


@dataclass(frozen=True)
class Pipeline:
    locator: Locator
    classifier: StateClassifier
    labeler: RapidOcrLabeler

    def run(self, image: np.ndarray, labels: bool = False) -> list[Box]:
        rects = self.locator.locate(image)
        states = self.classifier.classify(image, rects)
        boxes = sort_reading_order([Box(r, c, s) for r, (c, s) in zip(rects, states, strict=True)])

        if labels:
            texts = self.labeler.label(image, [b.rect for b in boxes])
            boxes = [replace(b, label=t) for b, t in zip(boxes, texts, strict=True)]
        return boxes


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
        RapidOcrLabeler(),
    )
