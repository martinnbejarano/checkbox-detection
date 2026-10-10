"""Stage 3: the label text next to each box. Must not import `locate`, `classify` or anything
outside `detection`.
"""

import numpy as np

from checkbox_detection.detection.types import Rect


class RapidOcrLabeler:
    # Stub: RapidOCR recognition arrives with the label ticket; until then no label is read.
    def label(self, image: np.ndarray, rects: list[Rect]) -> list[str | None]:
        return [None] * len(rects)
