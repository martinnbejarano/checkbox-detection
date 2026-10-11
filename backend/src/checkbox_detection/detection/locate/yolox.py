"""YoloxLocator: `nutrientdocs/form-field-v1-nano` run over overlapping tiles of the page.

Must not import `classify`, `label` or anything outside `detection`.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from checkbox_detection.detection.binarize import to_clean_gray
from checkbox_detection.detection.geometry import nms, tile_windows
from checkbox_detection.detection.onnx import load_session
from checkbox_detection.detection.types import Rect

MODEL_FILE = "form-field-v1-nano.onnx"
INPUT = 640  # the model's fixed square input
PAD = 114  # letterbox grey the model was trained with; pixels stay raw 0-255
CHOICE_BUTTON = 1  # classes: 0 Text, 1 ChoiceButton, 2 Signature


@dataclass(frozen=True, slots=True)
class YoloxParams:
    # Tile side as a fraction of the page width (1530 px on a 2550 px page); each tile is shrunk
    # to 640 px, so this sets how big a box looks to the model. Width, not the short edge, so a
    # wide crop is not cut into tiles that blow its boxes up. On the samples 0.5 loses large boxes
    # and 0.7 small ones; 0.6 finds 96-100% on all four.
    # ponytail: one tile size, so only boxes about 0.6-2.2% of the page width are seen (the
    # model wants 10-25 px boxes in its input); run a second tile size and merge if forms with
    # bigger or smaller boxes turn up.
    tile_frac: float = 0.6
    # Neighbouring tiles overlap by this fraction of the width, more than the largest allowed box
    # side, so every box is whole in at least one tile.
    overlap_frac: float = 0.08
    # objectness * ChoiceButton probability below this is dropped. Low, for recall: 0.3 already
    # loses a few real boxes on the cropped sample.
    score_threshold: float = 0.2
    # The same box found in two tiles overlaps more than this.
    nms_iou: float = 0.3
    # Size extremes as for the OpenCV locator, the upper one a fraction of the page width.
    min_side_px: int = 8
    max_side_frac: float = 0.06


class YoloxLocator:
    def __init__(self, models_dir: Path, threads: int) -> None:
        self.session = load_session(models_dir / MODEL_FILE, threads)
        self.params = YoloxParams()

    def locate(self, image: np.ndarray) -> list[Rect]:
        # Red watermarks and blue cell fills turned white, as for the OpenCV locator; the model
        # takes 3 channels.
        rgb = cv2.cvtColor(to_clean_gray(image), cv2.COLOR_GRAY2RGB)
        height, width = rgb.shape[:2]
        windows = tile_windows(
            width,
            height,
            tile=round(self.params.tile_frac * width),
            overlap=round(self.params.overlap_frac * width),
        )
        rects = [r for w in windows for r in self._locate_tile(rgb, w)]
        max_side = self.params.max_side_frac * width
        return [
            r
            for r in nms(rects, self.params.nms_iou)
            if self.params.min_side_px <= min(r.x2 - r.x1, r.y2 - r.y1)
            and max(r.x2 - r.x1, r.y2 - r.y1) <= max_side
        ]

    def _locate_tile(self, rgb: np.ndarray, window: tuple[int, int, int, int]) -> list[Rect]:
        x1, y1, x2, y2 = window
        tile = rgb[y1:y2, x1:x2]
        # Letterbox: scale the long side to 640, pad right and bottom.
        scale = INPUT / max(tile.shape[:2])
        size = (round(tile.shape[1] * scale), round(tile.shape[0] * scale))
        resized = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
        canvas = np.full((INPUT, INPUT, 3), PAD, np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None]

        # Rows are [cx, cy, w, h, objectness, p_text, p_choice, p_signature], in 640 px.
        out = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        scores = out[:, 4] * out[:, 5 + CHOICE_BUTTON]
        keep = (scores >= self.params.score_threshold) & (out[:, 5:].argmax(1) == CHOICE_BUTTON)
        return [
            Rect(
                round(x1 + (cx - w / 2) / scale),
                round(y1 + (cy - h / 2) / scale),
                round(x1 + (cx + w / 2) / scale),
                round(y1 + (cy + h / 2) / scale),
                float(score),
            )
            for (cx, cy, w, h), score in zip(out[keep, :4], scores[keep], strict=True)
        ]
