import cv2
import numpy as np
import pytest
from conftest import blank_page, draw_box

from checkbox_detection.detection.label import LabelParams, RapidOcrLabeler, label_strip
from checkbox_detection.detection.types import Rect


def write(img: np.ndarray, text: str, x: int, y: int) -> None:
    """Text about as tall as a 30 px box, with its baseline at y."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)


def test_reads_text_right_of_box() -> None:
    img = blank_page(1000, 400)
    draw_box(img, 100, 200, 30, "empty")
    write(img, "Urban", 140, 225)

    labels = RapidOcrLabeler(1).label(img, [Rect(100, 200, 130, 230, 1.0)])

    assert labels == ["Urban"]


@pytest.mark.parametrize("x", [100, 970], ids=["blank-strip", "page-edge"])
def test_nothing_right_of_box_is_none(x: int) -> None:
    img = blank_page(1000, 400)
    draw_box(img, x, 200, 30, "empty")

    labels = RapidOcrLabeler(1).label(img, [Rect(x, 200, x + 30, 230, 1.0)])

    assert labels == [None]


def test_strip_stops_at_next_box_on_row() -> None:
    img = blank_page(1000, 400)
    draw_box(img, 100, 200, 30, "empty")
    write(img, "Urban", 140, 225)
    draw_box(img, 240, 200, 30, "x")  # closer than the blank gap that also ends a label
    write(img, "Rural", 280, 225)
    rects = [Rect(100, 200, 130, 230, 1.0), Rect(240, 200, 270, 230, 1.0)]

    labels = RapidOcrLabeler(1).label(img, rects)

    assert labels == ["Urban", "Rural"]


BOX = Rect(100, 200, 130, 230, 1.0)  # 30 px side: 8 px band margin, 2 px start gap


@pytest.mark.parametrize(
    ("others", "page_width", "expected"),
    [
        ([], 2000, (132, 192, 580, 238)),  # 15 box sides
        ([Rect(240, 200, 270, 230, 1.0)], 2000, (132, 192, 240, 238)),  # next box on the row
        ([Rect(240, 240, 270, 270, 1.0)], 2000, (132, 192, 580, 238)),  # box on the next row
        ([], 300, (132, 192, 300, 238)),  # page edge
    ],
    ids=["max-width", "next-box", "other-row", "page-edge"],
)
def test_strip_geometry(others: list[Rect], page_width: int, expected: tuple[int, ...]) -> None:
    assert label_strip(BOX, [BOX, *others], page_width, 1000, LabelParams()) == expected
