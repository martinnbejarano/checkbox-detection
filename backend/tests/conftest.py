"""Helpers shared by test files."""

from pathlib import Path
from typing import Literal

import cv2
import httpx2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"  # filled by `make models`


def assert_problem(response: httpx2.Response, status: int) -> None:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body.keys() == {"type", "title", "status", "detail"}
    assert body["status"] == status
    assert "Traceback" not in response.text


def blank_page(width: int, height: int) -> np.ndarray:
    return np.full((height, width), 255, np.uint8)


def as_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


Mark = Literal["empty", "x", "check", "filled"]


def draw_box(img: np.ndarray, x: int, y: int, side: int, mark: Mark) -> None:
    """A black box of `side` px with its top-left at (x, y), holding `mark`."""
    t = max(1, side // 15)
    color = (0, 0, 0) if img.ndim == 3 else 0
    cv2.rectangle(img, (x, y), (x + side - 1, y + side - 1), color, t)
    a, b = x + side // 4, x + 3 * side // 4  # mark spans the middle half
    c, d = y + side // 4, y + 3 * side // 4
    if mark == "x":
        cv2.line(img, (a, c), (b, d), color, t)
        cv2.line(img, (b, c), (a, d), color, t)
    elif mark == "check":
        cv2.polylines(
            img, [np.array([(a, (c + d) // 2), ((a + b) // 2, d), (b, c)])], False, color, t
        )
    elif mark == "filled":
        cv2.rectangle(img, (a, c), (b, d), color, -1)
