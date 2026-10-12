from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import MODELS_DIR, blank_page, draw_box

from checkbox_detection.detection import build_pipeline

SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "sample-001.png"


@pytest.mark.parametrize(
    "turned_by",
    [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE],
    ids=["upright", "90", "180", "270"],
)
def test_turned_page_comes_back_upright(turned_by: int | None) -> None:
    page = cv2.imread(str(SAMPLE), cv2.IMREAD_GRAYSCALE)
    turned = page if turned_by is None else cv2.rotate(page, turned_by)

    upright = build_pipeline("opencv", "fill_ratio", MODELS_DIR, 1).straighten(turned)

    assert np.array_equal(upright.image, page)


def test_page_without_text_is_not_turned() -> None:
    page = blank_page(1000, 1400)
    for y in range(200, 1200, 100):
        draw_box(page, 150, y, 40, "x")

    upright = build_pipeline("opencv", "fill_ratio", MODELS_DIR, 1).straighten(page)

    assert upright.image is page
