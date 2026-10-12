import math

import cv2
import numpy as np
import pytest
from conftest import blank_page, draw_box

from checkbox_detection.detection.deskew import deskew


def form_page() -> np.ndarray:
    """A letter page at 200 dpi: rows of a box and its label, under a rule."""
    img = blank_page(1700, 2200)
    cv2.line(img, (150, 200), (1550, 200), 0, 3)
    for i, y in enumerate(range(300, 2000, 80)):
        draw_box(img, 150, y, 40, "x" if i % 3 == 0 else "empty")
        cv2.putText(
            img, f"Option {i} of the form", (210, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2
        )
    return img


@pytest.mark.parametrize("tilt", [-7.0, -2.0, 3.0, 8.0])
def test_finds_the_tilt_of_a_page(tilt: float) -> None:
    page = form_page()
    h, w = page.shape
    tilted = cv2.warpAffine(
        page, cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1), (w, h), borderValue=255
    )

    to_original = deskew(tilted).to_original

    # Rotation of the matrix back to the tilted page, in the convention of getRotationMatrix2D.
    assert math.degrees(math.atan2(to_original[0, 1], to_original[0, 0])) == pytest.approx(
        tilt, abs=0.25
    )


def test_upright_page_comes_back_untouched() -> None:
    page = form_page()

    upright = deskew(page)

    assert upright.image is page
    assert np.array_equal(upright.to_original, [[1, 0, 0], [0, 1, 0]])
