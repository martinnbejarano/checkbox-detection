"""Stage 1: where are the checkboxes. The `Locator` Protocol; implementations live next to it.

Must not import `classify`, `label` or anything outside `detection`.
"""

from typing import Protocol

import numpy as np

from checkbox_detection.detection.types import Rect


class Locator(Protocol):
    def locate(self, image: np.ndarray) -> list[Rect]: ...
