"""Gray conversion, binarization and ink measures shared by the OpenCV locator and fill_ratio.

Lives outside `locate/` and `classify/` so neither stage imports the other.
Must not import anything outside `detection`.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BinarizeParams:
    # Adaptive-threshold block as a fraction of page width, so it scales with resolution.
    block_frac: float = 0.02
    # A pixel is ink when it is this much darker than its neighbourhood mean.
    threshold_offset: int = 10


def to_clean_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def binarize(gray: np.ndarray, params: BinarizeParams | None = None) -> np.ndarray:
    """Ink is 255, paper is 0."""
    params = params or BinarizeParams()
    block = max(3, int(gray.shape[1] * params.block_frac) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        params.threshold_offset,
    )


def interior_ink(binary: np.ndarray, x: int, y: int, w: int, h: int, margin_frac: float) -> float:
    """Share of ink pixels inside the box, minus a `margin_frac` border."""
    mx, my = int(w * margin_frac), int(h * margin_frac)
    interior = binary[y + my : y + h - my, x + mx : x + w - mx]
    return float(np.mean(interior > 0)) if interior.size else 0.0
