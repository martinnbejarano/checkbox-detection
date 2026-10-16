"""Gray conversion, binarization and line removal shared by the OpenCV locator and fill_ratio.

Lives outside `locate/` and `classify/` so neither stage imports the other.
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
    # HSV saturation (0-255) above which a pixel is colour overlay (red watermark, blue fill).
    # Black and gray ink stay well below it.
    min_overlay_saturation: int = 70
    # Only pixels at least this bright (HSV value, 0-255) count as overlay: dark pixels have
    # noisy saturation and are ink.
    min_overlay_value: int = 100


def to_gray(image: np.ndarray) -> np.ndarray:
    """Plain grayscale. A gray image is returned unchanged."""
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_clean_gray(image: np.ndarray, params: BinarizeParams | None = None) -> np.ndarray:
    """Grayscale with colour overlays turned to paper. A gray image is returned unchanged."""
    if image.ndim == 2:
        return image
    params = params or BinarizeParams()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    overlay = (hsv[..., 1] > params.min_overlay_saturation) & (
        hsv[..., 2] >= params.min_overlay_value
    )
    gray = to_gray(image)
    gray[overlay] = 255
    return gray


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


def remove_lines(binary: np.ndarray, min_length: int) -> np.ndarray:
    """Erase horizontal and vertical runs of ink at least `min_length` px long (table rules)."""
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, min_length), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((min_length, 1), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)))


def border_thickness(box: np.ndarray) -> int:
    """Median ink run from each edge inward along the middle row and column."""
    h, w = box.shape
    lines = [box[h // 2], box[h // 2, ::-1], box[:, w // 2], box[::-1, w // 2]]
    runs = [int(np.argmin(line > 0)) if not line.all() else len(line) for line in lines]
    return max(1, min(int(np.median(runs)), int(min(h, w)) // 4))
