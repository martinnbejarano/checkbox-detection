"""StateCnnClassifier: `nutrientdocs/form-field-v1-state`, fine-tuned (training/train_cnn.py), on
40x160 gray crops, one batched run."""

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
    # Tuned on validation by training/train_cnn.py tune.
    # The crop reaches this share of the side past each box edge, so a border a locator cut a few
    # px short is still whole.
    pad_frac: float = 0.05
    # P(filled) above this means checked: the model's own 0.5 plus a 0.2 margin, so an ambiguous
    # score resolves to unchecked.
    threshold: float = 0.7


class StateCnnClassifier:
    def __init__(self, models_dir: Path, threads: int) -> None:
        self.session = load_session(models_dir / MODEL_FILE, threads)
        self.params = StateCnnParams()

    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]:
        if not rects:
            return []
        gray = to_clean_gray(image)
        crops = np.stack([crop(gray, r, self.params.pad_frac) for r in rects])[:, None]
        x = (crops.astype(np.float32) / 255 - MEAN) / STD
        logits = self.session.run(None, {self.session.get_inputs()[0].name: x})[0]
        filled = 1 / (1 + np.exp(logits[:, 0] - logits[:, 1]))  # softmax over (empty, filled)
        return [(bool(p > self.params.threshold), float(p)) for p in filled]


def crop(gray: np.ndarray, rect: Rect, pad_frac: float) -> np.ndarray:
    """The box padded by `pad_frac` of its side, letterboxed into 40x160 on white, centred, aspect
    kept. Training (training/train_cnn.py) makes its crops with it too."""
    pad = round(pad_frac * min(rect.x2 - rect.x1, rect.y2 - rect.y1))
    box = gray[max(0, rect.y1 - pad) : rect.y2 + pad, max(0, rect.x1 - pad) : rect.x2 + pad]
    scale = min(WIDTH / box.shape[1], HEIGHT / box.shape[0])
    w, h = max(1, round(box.shape[1] * scale)), max(1, round(box.shape[0] * scale))
    out = np.full((HEIGHT, WIDTH), 255, np.uint8)
    x0, y0 = (WIDTH - w) // 2, (HEIGHT - h) // 2
    out[y0 : y0 + h, x0 : x0 + w] = cv2.resize(box, (w, h), interpolation=cv2.INTER_LINEAR)
    return out
