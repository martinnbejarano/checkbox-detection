"""Stage 0: `Orienter` turns a page on its side or upside down upright, with RapidOCR text
lines, and levels its tilt (`deskew.py`)."""

from dataclasses import dataclass

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_det import TextDetector
from rapidocr.ch_ppocr_rec import TextRecInput, TextRecognizer

from checkbox_detection.detection.binarize import to_clean_gray
from checkbox_detection.detection.deskew import deskew
from checkbox_detection.detection.geometry import Upright


@dataclass(frozen=True, slots=True)
class OrientParams:
    # Text lines are detected on the page shrunk to this many px on its long side.
    work_px: int = 1600
    # With fewer text lines the page is not turned.
    min_lines: int = 5
    # The widest lines, recognized as they are and turned 180° to tell an upside-down page.
    probe_lines: int = 10
    # Each probe is cut to this many times its height: enough text, and a short batch to read.
    probe_aspect: float = 8.0


PARAMS = OrientParams()


class Orienter:
    """`ocr` is shared with stage 3 (`label.py`); only its models are called here."""

    def __init__(self, ocr: RapidOCR) -> None:
        # The models are called directly: `ocr(...)` keeps its use_det/use_rec flags on the shared
        # instance, which the labeler may be using at the same time. Each is loaded once, here,
        # and kept for the life of the pipeline.
        self.detector = ocr._load_det_model()  # type: ignore[no-untyped-call]
        self.recognizer = ocr._load_rec_model()  # type: ignore[no-untyped-call]

    def straighten(self, image: np.ndarray) -> Upright:
        """The stages take `upright.image`; `upright.boxes_to_original` maps their boxes back to
        `image`. Quarter turn, then deskew, then half turn: upside down is told apart on a level
        page only."""
        lines = text_lines(image, self.detector)
        upright = quarter_turn(image) if is_sideways(lines) else Upright.identity(image)
        upright = upright.then(deskew(upright.image))
        if is_upside_down(upright, lines, self.recognizer):
            upright = upright.then(half_turn(upright.image))
        return upright


def quarter_turn(image: np.ndarray) -> Upright:
    """The page turned 90° clockwise."""
    h, w = image.shape[:2]  # turned (x', y') -> original (x, y) = (y', h - x')
    return Upright(
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), np.array([[0.0, 1, 0], [-1, 0, h]]), (w, h)
    )


def half_turn(image: np.ndarray) -> Upright:
    """The page turned 180°."""
    h, w = image.shape[:2]
    return Upright(cv2.rotate(image, cv2.ROTATE_180), np.array([[-1.0, 0, w], [0, -1, h]]), (w, h))


def text_lines(image: np.ndarray, detector: TextDetector) -> np.ndarray:
    """Corners of each text line, (n, 4, 2) in `image` pixels."""
    gray = to_clean_gray(image)
    scale = PARAMS.work_px / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = detector(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return np.zeros((0, 4, 2)) if out.boxes is None else out.boxes / min(scale, 1)


def is_sideways(lines: np.ndarray) -> bool:
    """Most lines are taller than wide."""
    if len(lines) < PARAMS.min_lines:
        return False
    w, h = np.ptp(lines[..., 0], axis=1), np.ptp(lines[..., 1], axis=1)
    return bool((h > w).mean() > 0.5)


def is_upside_down(level: Upright, lines: np.ndarray, recognizer: TextRecognizer) -> bool:
    """The widest `lines` (original page pixels) read better turned 180° on `level.image`."""
    if len(lines) < PARAMS.min_lines:
        return False
    m = cv2.invertAffineTransform(level.to_original)
    lines = lines @ m[:, :2].T + m[:, 2]
    widest = lines[np.argsort(-np.ptp(lines[..., 0], axis=1))[: PARAMS.probe_lines]]
    h, w = level.image.shape[:2]
    probes = []
    for x1, y1, x2, y2 in np.concatenate([widest.min(axis=1), widest.max(axis=1)], axis=1):
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2 = min(int(x2) + 1, w, x1 + round(PARAMS.probe_aspect * (y2 - y1)))
        crop = to_clean_gray(level.image[y1 : min(int(y2) + 1, h), x1:x2])
        probes.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    turned = [cv2.rotate(p, cv2.ROTATE_180) for p in probes]
    scores = recognizer(TextRecInput(img=probes + turned)).scores
    n = len(probes)
    return bool(np.mean(scores[n:]) > np.mean(scores[:n]))
