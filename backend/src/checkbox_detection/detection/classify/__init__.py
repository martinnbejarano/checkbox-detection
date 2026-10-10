"""Stage 2: is each checkbox checked. The `StateClassifier` Protocol; implementations sit beside it.

Must not import `locate`, `label` or anything outside `detection`.
"""

from typing import Protocol

import numpy as np

from checkbox_detection.detection.types import Rect


class StateClassifier(Protocol):
    # Batched: all rects of a page in one call, so an ONNX model does one session.run.
    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]: ...
