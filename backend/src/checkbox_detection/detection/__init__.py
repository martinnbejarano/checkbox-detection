"""Checkbox detection on one decoded page: pure NumPy/OpenCV, no I/O.

Must not import FastAPI, `api`, `imaging` or `config`; only its own modules.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

from checkbox_detection.detection.opencv.detector import OpenCVDetector
from checkbox_detection.detection.types import Box


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]: ...


DETECTORS: dict[str, Callable[[], Detector]] = {"opencv": OpenCVDetector}


def build_detector(name: str) -> Detector:
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}; choose one of {sorted(DETECTORS)}")
    return DETECTORS[name]()
