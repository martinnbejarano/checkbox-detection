"""Stage 0: straighten a page tilted up to a few degrees, by projection profile."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from checkbox_detection.detection.binarize import binarize, to_clean_gray
from checkbox_detection.detection.geometry import Upright


@dataclass(frozen=True, slots=True)
class DeskewParams:
    # The angle is searched on the page shrunk to this many px on its long side.
    work_px: int = 1000
    # Tilts tried, in degrees: -max_angle..+max_angle in `step` increments.
    max_angle: float = 10.0
    step: float = 0.25
    # Below this tilt, in degrees, the page is returned as is (no warp).
    min_angle: float = 0.3


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
    # The angle is the same at any size; searching it small is faster.
    scale = params.work_px / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    angle = skew_angle(binarize(gray), params)
    if abs(angle) < params.min_angle:
        return Upright.identity(image)
    rotated, m = rotate_whole(image, angle)
    return Upright(rotated, cv2.invertAffineTransform(m), (w, h))


def rotate_whole(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """`image` rotated `angle` degrees on a canvas enlarged to keep every corner, filled with
    white; and the 2x3 affine used."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cos, sin = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    new_w, new_h = math.ceil(w * cos + h * sin), math.ceil(w * sin + h * cos)
    # Shift so the page is centred on the larger canvas.
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    white = (255,) * (image.shape[2] if image.ndim == 3 else 1)
    rotated = cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=white)
    return rotated, m
