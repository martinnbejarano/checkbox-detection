"""Stage 3: the label text next to each box. Must not import `locate`, `classify` or anything
outside `detection`.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from rapidocr import RapidOCR

from checkbox_detection.detection.binarize import remove_lines, to_clean_gray
from checkbox_detection.detection.types import Rect


@dataclass(frozen=True, slots=True)
class LabelParams:
    # Added above and below the box for the row band, as a share of the side.
    band_margin_frac: float = 0.25
    # Skipped right of the box so its edge is not read, as a share of the side.
    start_gap_frac: float = 0.05
    # Longest strip, in box sides; long enough for "Legal Nonconforming (Grandfathered Use)".
    max_width_sides: int = 15
    # Blank paper after the text, in box sides, that ends the label.
    end_gap_sides: float = 1.0
    # Gray level below which a pixel is ink.
    ink_below: int = 128
    # A column inked over at least this share of the band is a vertical rule; it ends the label.
    rule_frac: float = 0.9
    # White margin left and right of the strip, as a share of its height: the recognizer drops
    # a letter that touches the crop's edge.
    pad_frac: float = 0.25
    # Recognition confidence below which the text is dropped.
    min_score: float = 0.5


def label_strip(
    rect: Rect, rects: list[Rect], width: int, height: int, params: LabelParams
) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) of the strip right of `rect`: its row band, up to the next box on the
    row, a width proportional to the box side, or the page edge. Empty when x2 <= x1."""
    side = max(rect.x2 - rect.x1, rect.y2 - rect.y1)
    margin = round(params.band_margin_frac * side)
    x1 = rect.x2 + round(params.start_gap_frac * side)
    cy = (rect.y1 + rect.y2) / 2
    next_box = min((r.x1 for r in rects if r.x1 >= rect.x2 and r.y1 <= cy <= r.y2), default=width)
    x2 = min(rect.x2 + params.max_width_sides * side, next_box, width)
    return x1, max(0, rect.y1 - margin), x2, min(height, rect.y2 + margin)


def label_width(strip: np.ndarray, side: int, params: LabelParams) -> int:
    """Width of the label at the start of a gray strip: up to a vertical rule or a blank run of
    `end_gap_sides` after the text. 0 when the strip has no text."""
    if strip.size == 0:
        return 0
    ink = (strip < params.ink_below).astype(np.uint8) * 255
    rule = (ink > 0).sum(axis=0) >= params.rule_frac * strip.shape[0]
    text = remove_lines(ink, side).any(axis=0)  # underlines and row rules are not text
    end_gap = max(1, round(params.end_gap_sides * side))
    end, blank, seen = len(text), 0, False
    for x in range(len(text)):
        if rule[x]:
            end = x
            break
        seen = seen or bool(text[x])
        blank = 0 if text[x] else blank + 1
        if seen and blank >= end_gap:
            end = x - blank + 1
            break
    return end if text[:end].any() else 0


class RapidOcrLabeler:
    """RapidOCR recognition only (no text detection) on one strip per box. `ocr` is shared with
    stage 0 (`orient.py`)."""

    def __init__(self, threads: int) -> None:
        self.params = LabelParams()
        self.ocr = RapidOCR(
            params={
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.log_level": "warning",
                "Rec.rec_batch_num": 20,  # orient.py reads its 2 x 10 probes in one batch
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
        # RapidOCR loads its models on the first call; do it now, not on the first request. The
        # detection model is for `orient.py`, which shares this instance.
        self._read(np.full((32, 320), 255, np.uint8))
        self.ocr._load_det_model()  # type: ignore[no-untyped-call]

    # TODO(prod): labels left of the box, and the question of the row the box answers.
    def label(self, image: np.ndarray, rects: list[Rect]) -> list[str | None]:
        gray = to_clean_gray(image)
        h, w = gray.shape
        labels: list[str | None] = []
        for r in rects:
            x1, y1, x2, y2 = label_strip(r, rects, w, h, self.params)
            strip = gray[y1:y2, x1 : max(x1, x2)]
            width = label_width(strip, max(r.x2 - r.x1, r.y2 - r.y1), self.params)
            labels.append(self._read(strip[:, :width]) if width else None)
        return labels

    def _read(self, strip: np.ndarray) -> str | None:
        pad = round(self.params.pad_frac * strip.shape[0])
        strip = cv2.copyMakeBorder(strip, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)
        out = self.ocr(strip, use_det=False, use_cls=False, use_rec=True)
        text = out.txts[0].strip() if out.txts else ""  # type: ignore[union-attr]
        score = out.scores[0] if out.scores else 0.0  # type: ignore[union-attr]
        return text if text and score >= self.params.min_score else None
