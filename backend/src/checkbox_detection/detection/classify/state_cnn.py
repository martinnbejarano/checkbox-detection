"""StateCnnClassifier: `nutrientdocs/form-field-v1-state`, fine-tuned (training/train_cnn.py), on
40x160 gray crops, one batched run.

Must not import `locate`, `label` or anything outside `detection`.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from checkbox_detection.detection.binarize import to_clean_gray
from checkbox_detection.detection.onnx import load_session
from checkbox_detection.detection.types import Rect

MODEL_FILE = "form-field-v1-state-tuned.onnx"
HEIGHT, WIDTH = 40, 160  # the model's input crop
MEAN, STD = 0.5, 0.5  # the model's input normalisation, on pixels scaled to 0-1


@dataclass(frozen=True, slots=True)
class StateCnnParams:
    # Checked on the validation split, true boxes (training/train_cnn.py tune): state accuracy
    # 0.982 with these, against 0.955 for the pretrained model at its best setting. Boxes whose
    # mark the clean gray erases (colour ticks) are left out of both. Neither value moved: no
    # other candidate gains 0.5 points (the rule in training/search.py).
    # The crop reaches this share of the side past each box edge, so a border a locator cut a few
    # px short is still whole (0 to 0.15 tried, within 0.6 points).
    pad_frac: float = 0.05
    # P(filled) above this means checked: the model's own 0.5 plus a 0.2 margin, so an ambiguous
    # score resolves to unchecked (0.5 to 0.95 tried, all within 0.4 points).
    threshold: float = 0.7


class StateCnnClassifier:
    def __init__(self, models_dir: Path, threads: int) -> None:
        self.session = load_session(models_dir / MODEL_FILE, threads)
        self.params = StateCnnParams()

    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]:
        if not rects:
            return []
        gray = to_clean_gray(image)
        crops = np.stack([self._crop(gray, r) for r in rects])[:, None]
        x = (crops.astype(np.float32) / 255 - MEAN) / STD
        logits = self.session.run(None, {self.session.get_inputs()[0].name: x})[0]
        filled = 1 / (1 + np.exp(logits[:, 0] - logits[:, 1]))  # softmax over (empty, filled)
        return [(bool(p > self.params.threshold), float(p)) for p in filled]

    def _crop(self, gray: np.ndarray, rect: Rect) -> np.ndarray:
        """The padded box, letterboxed into 40x160 on white, centred, aspect kept."""
        pad = round(self.params.pad_frac * min(rect.x2 - rect.x1, rect.y2 - rect.y1))
        crop = gray[max(0, rect.y1 - pad) : rect.y2 + pad, max(0, rect.x1 - pad) : rect.x2 + pad]
        scale = min(WIDTH / crop.shape[1], HEIGHT / crop.shape[0])
        w, h = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
        out = np.full((HEIGHT, WIDTH), 255, np.uint8)
        x0, y0 = (WIDTH - w) // 2, (HEIGHT - h) // 2
        out[y0 : y0 + h, x0 : x0 + w] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return out
