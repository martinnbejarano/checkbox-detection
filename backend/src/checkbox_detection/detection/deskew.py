"""Stage 0: straighten a page tilted up to a few degrees, by projection profile.

Must not import anything outside `detection`.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from checkbox_detection.detection.binarize import binarize, to_clean_gray
from checkbox_detection.detection.geometry import map_rect
from checkbox_detection.detection.types import Box


@dataclass(frozen=True, slots=True)
class DeskewParams:
    # The angle is searched on the page shrunk to this many px on its long side.
    work_px: int = 1000
    # Tilts tried, in degrees: -max_angle..+max_angle in `step` increments.
    max_angle: float = 10.0
    step: float = 0.25
    # Below this tilt, in degrees, the page is returned as is (no warp).
    min_angle: float = 0.3


@dataclass(frozen=True, slots=True)
class Upright:
    image: np.ndarray  # the page as the stages see it
    to_original: np.ndarray  # 2x3 affine, upright pixels -> original pixels
    original_size: tuple[int, int]  # (width, height) of the original page, to clamp boxes

    @classmethod
    def identity(cls, image: np.ndarray) -> "Upright":
        """`image` as it is: no step."""
        return cls(image, np.eye(2, 3), (image.shape[1], image.shape[0]))

    def boxes_to_original(self, boxes: list[Box]) -> list[Box]:
        """Each box as the axis-aligned box around its mapped corners; the order is kept."""
        w, h = self.original_size
        return [replace(b, rect=map_rect(b.rect, self.to_original, w, h)) for b in boxes]

    def then(self, step: "Upright") -> "Upright":
        """`step`, computed on `self.image`, as one step from the original page."""
        to_original = self.to_original @ np.vstack([step.to_original, [0, 0, 1]])
        return Upright(step.image, to_original, self.original_size)


def skew_angle(binary: np.ndarray, params: DeskewParams) -> float:
    """Rotation (degrees, `cv2.getRotationMatrix2D` convention) that levels the rows of ink.

    Text lines and rules give peaky row sums when level and a flat profile when tilted, so the
    angle with the highest row-sum variance wins. Angles are tried from 0 outward and only a
    strictly higher variance replaces the best, so a blank page stays at 0.
    """
    h, w = binary.shape
    n = round(params.max_angle / params.step)
    best, best_var = 0.0, -1.0
    for angle in [0.0] + [s * i * params.step for i in range(1, n + 1) for s in (1, -1)]:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rows = cv2.reduce(cv2.warpAffine(binary, m, (w, h)), 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        var = float(rows.var())
        if var > best_var:
            best, best_var = angle, var
    return best


def deskew(image: np.ndarray) -> Upright:
    """The page rotated level on a canvas enlarged to keep every corner, filled with white."""
    params = DeskewParams()
    h, w = image.shape[:2]
    gray = to_clean_gray(image)
    scale = params.work_px / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    angle = skew_angle(binarize(gray), params)
    if abs(angle) < params.min_angle:
        return Upright.identity(image)

    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cos, sin = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    new_w, new_h = math.ceil(w * cos + h * sin), math.ceil(w * sin + h * cos)
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    white = (255,) * (image.shape[2] if image.ndim == 3 else 1)
    rotated = cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=white)
    return Upright(rotated, cv2.invertAffineTransform(m), (w, h))
