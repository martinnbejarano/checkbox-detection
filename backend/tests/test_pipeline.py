"""`Pipeline.detect` wires the stages; fakes stand in for the models, so only that is tested."""

import numpy as np
from conftest import blank_page

from checkbox_detection.detection import Pipeline
from checkbox_detection.detection.geometry import Upright
from checkbox_detection.detection.orient import half_turn
from checkbox_detection.detection.types import Box, Rect

# Upright pixels of a 200x100 page, out of reading order.
RECTS = [Rect(100, 10, 120, 30, 1.0), Rect(10, 60, 30, 80, 1.0), Rect(10, 10, 30, 30, 1.0)]


class FakeOrienter:
    def __init__(self, turn: bool) -> None:
        self.turn = turn

    def straighten(self, image: np.ndarray) -> Upright:
        return half_turn(image) if self.turn else Upright.identity(image)


class FakeLocator:
    def locate(self, image: np.ndarray) -> list[Rect]:
        return RECTS


class FakeClassifier:
    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]:
        return [(r.x1 == 100, 1.0) for r in rects]


class FakeLabeler:
    def __init__(self) -> None:
        self.calls = 0

    def label(self, image: np.ndarray, rects: list[Rect]) -> list[str | None]:
        self.calls += 1
        return [f"{r.x1},{r.y1}" for r in rects]


def pipeline(turn: bool, labeler: FakeLabeler) -> Pipeline:
    return Pipeline(FakeOrienter(turn), FakeLocator(), FakeClassifier(), labeler)  # type: ignore[arg-type]


def test_boxes_in_reading_order_without_labels() -> None:
    labeler = FakeLabeler()

    boxes, label_s = pipeline(False, labeler).detect(blank_page(200, 100))

    assert boxes == [
        Box(Rect(10, 10, 30, 30, 1.0), False),
        Box(Rect(100, 10, 120, 30, 1.0), True),
        Box(Rect(10, 60, 30, 80, 1.0), False),
    ]
    assert labeler.calls == 0
    assert label_s == 0.0


def test_labels_read_upright_then_boxes_mapped_back() -> None:
    boxes, _ = pipeline(True, FakeLabeler()).detect(blank_page(200, 100), labels=True)

    # Half turn: upright (x, y) is original (200 - x, 100 - y); labels keep the upright corner.
    assert boxes == [
        Box(Rect(170, 70, 190, 90, 1.0), False, "10,10"),
        Box(Rect(80, 70, 100, 90, 1.0), True, "100,10"),
        Box(Rect(170, 20, 190, 40, 1.0), False, "10,60"),
    ]
