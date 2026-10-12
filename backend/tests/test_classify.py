from collections.abc import Callable

import cv2
import numpy as np
import pytest
from conftest import MODELS_DIR, as_rgb, blank_page, draw_box

from checkbox_detection.detection.classify.fill_ratio import FillRatioClassifier
from checkbox_detection.detection.classify.state_cnn import StateCnnClassifier
from checkbox_detection.detection.types import Rect


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
@pytest.mark.parametrize("side", [12, 40, 90], ids=["12px", "40px", "90px"])
def test_states_per_mark_class(side: int, to_input: Callable[[np.ndarray], np.ndarray]) -> None:
    img = blank_page(2550, 3300)
    y = 1000
    xs = [300 + i * 4 * side for i in range(5)]
    for x, mark in zip(xs, ["empty", "x", "check", "filled", "empty"], strict=True):
        draw_box(img, x, y, side, mark)
    # A stray stroke across the last box, mostly outside it: not a mark.
    stray_x = xs[4]
    cv2.line(img, (stray_x - 2 * side, y + 2 * side), (stray_x + 3 * side, y - side), 0, 2)
    rects = [Rect(x, y, x + side, y + side, 1.0) for x in xs]

    states = FillRatioClassifier().classify(to_input(img), rects)

    assert [checked for checked, _ in states] == [False, True, True, True, False]
    assert states[4][1] == 0.0  # the stray stroke adds no ink, not just less than the threshold


def test_watermark_ink_is_not_a_mark() -> None:
    img = as_rgb(blank_page(2550, 3300))
    draw_box(img, 300, 1000, 50, "empty")
    cv2.line(img, (310, 1010), (340, 1040), (220, 40, 40), 6)  # red stroke inside the box
    cv2.line(img, (100, 1200), (600, 800), (220, 40, 40), 25)  # red diagonal band across it

    states = FillRatioClassifier().classify(img, [Rect(300, 1000, 350, 1050, 1.0)])

    assert [checked for checked, _ in states] == [False]


def test_near_threshold_fill_is_unchecked() -> None:
    img = blank_page(2550, 3300)
    draw_box(img, 300, 1000, 60, "empty")
    cv2.rectangle(img, (325, 1025), (333, 1033), 0, -1)  # 9x9 dot: ~6% of the interior

    states = FillRatioClassifier().classify(img, [Rect(300, 1000, 360, 1060, 1.0)])

    assert states[0][0] is False
    assert 0.05 < states[0][1] < 0.07  # inside the ambiguous band, not plain empty


# The CNN has no stray-stroke handling: a stroke crossing an empty box reads as checked.
@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
@pytest.mark.parametrize("side", [20, 40, 90], ids=["20px", "40px", "90px"])
def test_cnn_states_per_mark_class(side: int, to_input: Callable[[np.ndarray], np.ndarray]) -> None:
    img = blank_page(2550, 3300)
    y = 1000
    xs = [300 + i * 4 * side for i in range(4)]
    for x, mark in zip(xs, ["empty", "x", "check", "filled"], strict=True):
        draw_box(img, x, y, side, mark)
    rects = [Rect(x, y, x + side, y + side, 1.0) for x in xs]

    states = StateCnnClassifier(MODELS_DIR, 1).classify(to_input(img), rects)

    assert [checked for checked, _ in states] == [False, True, True, True]


def test_cnn_ambiguous_score_is_unchecked() -> None:
    img = blank_page(2550, 3300)
    draw_box(img, 300, 1000, 60, "empty")
    cv2.rectangle(img, (322, 1021), (337, 1039), 0, -1)  # 16x19 dot: P(filled) about 0.65

    states = StateCnnClassifier(MODELS_DIR, 1).classify(img, [Rect(300, 1000, 360, 1060, 1.0)])

    assert states[0][0] is False
    assert 0.5 < states[0][1] < 0.7  # inside the ambiguous band, not plain empty


def test_cnn_page_without_boxes_gives_no_states() -> None:
    assert StateCnnClassifier(MODELS_DIR, 1).classify(blank_page(2550, 3300), []) == []
