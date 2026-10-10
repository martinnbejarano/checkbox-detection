from collections.abc import Callable

import cv2
import numpy as np
import pytest
from conftest import as_rgb, blank_page, draw_box

from checkbox_detection.detection.locate.opencv import OpenCVLocator
from checkbox_detection.detection.types import Rect


def assert_boxes(found: list[Rect], expected: list[tuple[int, int, int, int]], side: int) -> None:
    """Same number of boxes, each bbox within +-3 px or +-5% of the side, whichever is larger."""
    tol = max(3, 0.05 * side)
    got = sorted((r.x1, r.y1, r.x2, r.y2) for r in found)
    assert len(got) == len(expected), got
    assert np.allclose(got, sorted(expected), atol=tol), got


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
@pytest.mark.parametrize(
    ("width", "side"),
    [(800, 12), (1000, 10), (1000, 50), (2500, 20), (2500, 100), (5000, 12), (5000, 100)],
    ids=["800-12-upscaled", "1000-10", "1000-50", "2500-20", "2500-100", "5000-12", "5000-100"],
)
def test_finds_every_mark_class_at_any_scale(
    width: int, side: int, to_input: Callable[[np.ndarray], np.ndarray]
) -> None:
    img = blank_page(width, int(width * 1.3))
    expected = []
    for i, mark in enumerate(["empty", "x", "check", "filled", "empty", "x"]):
        x, y = width // 10 + i * 3 * side, width // 3
        draw_box(img, x, y, side, mark)
        expected.append((x, y, x + side, y + side))

    found = OpenCVLocator().locate(to_input(img))

    assert_boxes(found, expected, side)


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
def test_finds_boxes_touching_table_lines(to_input: Callable[[np.ndarray], np.ndarray]) -> None:
    img = blank_page(2550, 3300)
    cv2.line(img, (0, 998), (2549, 998), 0, 3)  # rule resting on the row's top edges
    cv2.line(img, (0, 1051), (2549, 1051), 0, 3)  # rule under the row's bottom edges
    cv2.line(img, (751, 0), (751, 3299), 0, 3)  # rule against the second box's right edge
    for x, mark in [(300, "empty"), (700, "x"), (1100, "filled")]:
        draw_box(img, x, 1000, 50, mark)
    expected = [(300, 1000, 350, 1050), (700, 1000, 750, 1050), (1100, 1000, 1150, 1050)]

    found = OpenCVLocator().locate(to_input(img))

    assert_boxes(found, expected, 50)


def test_ignores_colour_overlays() -> None:
    img = as_rgb(blank_page(2550, 3300))
    cv2.rectangle(img, (400, 900), (1400, 1200), (150, 190, 230), -1)  # light-blue cell fill
    draw_box(img, 500, 1000, 50, "x")  # the only black box, sitting on the blue fill
    for x in range(300, 2000, 300):  # red boxes and a thick red diagonal watermark
        cv2.rectangle(img, (x, 2000), (x + 50, 2050), (220, 40, 40), 3)
    cv2.line(img, (200, 3000), (2400, 1500), (220, 40, 40), 25)

    found = OpenCVLocator().locate(img)

    assert_boxes(found, [(500, 1000, 550, 1050)], 50)


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
def test_text_only_page_has_no_boxes(to_input: Callable[[np.ndarray], np.ndarray]) -> None:
    img = blank_page(2550, 3300)
    for i, text in enumerate(["DO 0 O D Q", "Owner Tenant Vacant", "PUD 00 DD OO"]):
        cv2.putText(img, text, (200, 500 + i * 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)

    assert OpenCVLocator().locate(to_input(img)) == []
