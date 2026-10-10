from collections.abc import Callable
from functools import cache

import cv2
import numpy as np
import pytest
from conftest import MODELS_DIR, as_rgb, blank_page, draw_box

from checkbox_detection.detection import LOCATORS
from checkbox_detection.detection.locate import Locator
from checkbox_detection.detection.types import Rect


@cache
def locator(name: str) -> Locator:
    return LOCATORS[name](MODELS_DIR, 1)


# A bbox may be off by 3 px or this share of the side, whichever is larger. YOLOX sees the page
# shrunk to 640 px tiles, so one px of its error is several page px.
SIDE_TOLERANCE = {"opencv": 0.05, "yolox": 0.08}


def assert_boxes(
    found: list[Rect], expected: list[tuple[int, int, int, int]], side: int, name: str
) -> None:
    """Same number of boxes, and each expected bbox found within the locator's tolerance."""
    tol = max(3, SIDE_TOLERANCE[name] * side)
    got = [(r.x1, r.y1, r.x2, r.y2) for r in found]
    assert len(got) == len(expected), got
    for box in expected:
        assert any(np.allclose(g, box, atol=tol) for g in got), (box, got)


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
@pytest.mark.parametrize(
    ("name", "width", "side"),
    [
        ("opencv", 800, 12),  # upscaled
        ("opencv", 1000, 10),
        ("opencv", 1000, 50),
        ("opencv", 2500, 20),
        ("opencv", 2500, 100),
        ("opencv", 5000, 12),
        ("opencv", 5000, 100),
        # The model sees a box only at about 0.6-2.2% of the page width (see YoloxParams).
        ("yolox", 1000, 8),
        ("yolox", 1000, 20),
        ("yolox", 2500, 15),
        ("yolox", 2500, 45),
        ("yolox", 5000, 30),
        ("yolox", 5000, 100),
    ],
    ids=lambda v: str(v),
)
def test_finds_every_mark_class_at_any_scale(
    name: str, width: int, side: int, to_input: Callable[[np.ndarray], np.ndarray]
) -> None:
    img = blank_page(width, int(width * 1.3))
    expected = []
    for i, mark in enumerate(["empty", "x", "check", "filled", "empty", "x"]):
        x, y = width // 10 + (i % 3) * 6 * side, width // 3 + (i // 3) * 2 * side
        draw_box(img, x, y, side, mark)
        org = (x + side + side // 3, y + side - side // 6)  # a label right of the box
        cv2.putText(img, "Yes", org, cv2.FONT_HERSHEY_SIMPLEX, side / 30, 0, max(1, side // 15))
        expected.append((x, y, x + side, y + side))

    found = locator(name).locate(to_input(img))

    assert_boxes(found, expected, side, name)


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
def test_opencv_finds_boxes_touching_table_lines(
    to_input: Callable[[np.ndarray], np.ndarray],
) -> None:
    img = blank_page(2550, 3300)
    cv2.line(img, (0, 998), (2549, 998), 0, 3)  # rule resting on the row's top edges
    cv2.line(img, (0, 1051), (2549, 1051), 0, 3)  # rule under the row's bottom edges
    cv2.line(img, (751, 0), (751, 3299), 0, 3)  # rule against the second box's right edge
    for x, mark in [(300, "empty"), (700, "x"), (1100, "filled")]:
        draw_box(img, x, 1000, 50, mark)
    expected = [(300, 1000, 350, 1050), (700, 1000, 750, 1050), (1100, 1000, 1150, 1050)]

    found = locator("opencv").locate(to_input(img))

    assert_boxes(found, expected, 50, "opencv")


def test_opencv_ignores_colour_overlays() -> None:
    img = as_rgb(blank_page(2550, 3300))
    cv2.rectangle(img, (400, 900), (1400, 1200), (150, 190, 230), -1)  # light-blue cell fill
    draw_box(img, 500, 1000, 50, "x")  # the only black box, sitting on the blue fill
    for x in range(300, 2000, 300):  # red boxes and a thick red diagonal watermark
        cv2.rectangle(img, (x, 2000), (x + 50, 2050), (220, 40, 40), 3)
    cv2.line(img, (200, 3000), (2400, 1500), (220, 40, 40), 25)

    found = locator("opencv").locate(img)

    assert_boxes(found, [(500, 1000, 550, 1050)], 50, "opencv")


def test_yolox_ignores_red_overlays() -> None:
    img = as_rgb(blank_page(2550, 3300))
    for x in (360, 600, 840):  # two rows of black labelled boxes
        for y in (1000, 1200):
            draw_box(img, x, y, 40, "x")
            cv2.putText(img, "Yes", (x + 55, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        # and one row of red ones under a red diagonal watermark
        cv2.rectangle(img, (x, 2000), (x + 40, 2040), (220, 40, 40), 3)
        cv2.putText(img, "Yes", (x + 55, 2033), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (220, 40, 40), 2)
    cv2.line(img, (200, 3000), (2400, 1500), (220, 40, 40), 25)
    expected = [(x, y, x + 40, y + 40) for x in (360, 600, 840) for y in (1000, 1200)]

    found = locator("yolox").locate(img)

    assert_boxes(found, expected, 40, "yolox")


@pytest.mark.parametrize("to_input", [lambda g: g, as_rgb], ids=["gray", "rgb"])
@pytest.mark.parametrize("name", ["opencv", "yolox"])
def test_text_only_page_has_no_boxes(
    name: str, to_input: Callable[[np.ndarray], np.ndarray]
) -> None:
    img = blank_page(2550, 3300)
    for i, text in enumerate(["DO 0 O D Q", "Owner Tenant Vacant", "PUD 00 DD OO"]):
        cv2.putText(img, text, (200, 500 + i * 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)

    assert locator(name).locate(to_input(img)) == []


def test_yolox_finds_a_box_across_tile_borders_once() -> None:
    # On a 2550x3300 page the tiles are 1530 px, starting at x 0/1020 and y 0/885/1770. Each row
    # of labelled boxes has its middle box across a border: x=1530, x=y=1530, and y=2415.
    img = blank_page(2550, 3300)
    expected = []
    for xs, y in [((1270, 1510, 1750), 600), ((1270, 1510, 1750), 1510), ((360, 600, 840), 2400)]:
        for x in xs:
            draw_box(img, x, y, 40, "x")
            cv2.putText(img, "Yes", (x + 55, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 2)
            expected.append((x, y, x + 40, y + 40))

    found = locator("yolox").locate(img)

    assert_boxes(found, expected, 40, "yolox")
