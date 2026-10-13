from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import blank_page, draw_box

from checkbox_detection.detection.ocr import load_ocr
from checkbox_detection.detection.orient import Orienter

SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "sample-001.png"


@pytest.fixture(scope="module")
def orienter() -> Orienter:
    return Orienter(load_ocr(1))


@pytest.fixture(scope="module")
def page() -> np.ndarray:
    return cv2.imread(str(SAMPLE), cv2.IMREAD_GRAYSCALE)


@pytest.mark.parametrize(
    "turned_by",
    [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE],
    ids=["90", "180", "270"],
)
def test_turned_page_comes_back_upright(
    orienter: Orienter, page: np.ndarray, turned_by: int
) -> None:
    upright = orienter.straighten(cv2.rotate(page, turned_by))

    assert np.array_equal(upright.image, page)


def test_upright_page_is_not_turned(orienter: Orienter, page: np.ndarray) -> None:
    upright = orienter.straighten(page)

    assert np.array_equal(upright.image, page)


def test_page_without_text_is_not_turned(orienter: Orienter) -> None:
    page = blank_page(1000, 1400)
    for y in range(200, 1200, 100):
        draw_box(page, 150, y, 40, "x")

    upright = orienter.straighten(page)

    assert upright.image is page
