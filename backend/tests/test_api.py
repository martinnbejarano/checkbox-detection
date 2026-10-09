import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from checkbox_detection.api.routes import get_detector
from checkbox_detection.config import Settings
from checkbox_detection.detection.types import Box
from checkbox_detection.main import create_app


def png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def two_box_page() -> bytes:
    """A white page with an empty box on the left and an X box on the right."""
    page = np.full((1000, 1000), 255, np.uint8)
    cv2.rectangle(page, (200, 400), (250, 450), 0, 3)
    cv2.rectangle(page, (600, 400), (650, 450), 0, 3)
    cv2.line(page, (608, 408), (642, 442), 0, 4)
    cv2.line(page, (642, 408), (608, 442), 0, 4)
    return png_bytes(page)


def test_detect_returns_empty_and_checked_box() -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.post("/detect", files={"file": ("page.png", two_box_page())})

    assert response.status_code == 200
    boxes = response.json()["boxes"]
    assert [b["is_checked"] for b in boxes] == [False, True]
    assert [b["page"] for b in boxes] == [0, 0]
    for box, expected in zip(boxes, ([200, 400, 250, 450], [600, 400, 650, 450]), strict=True):
        assert np.allclose(box["bbox"], expected, atol=3)


def test_detect_returns_contract_shape_from_any_detector() -> None:
    class FakeDetector:
        def detect(self, image: np.ndarray) -> list[Box]:
            return [Box(1, 2, 3, 4, is_checked=True, score=0.9)]

    app = create_app(Settings())
    app.dependency_overrides[get_detector] = FakeDetector
    with TestClient(app) as client:
        response = client.post("/detect", files={"file": ("page.png", two_box_page())})

    assert response.status_code == 200
    assert response.json() == {"boxes": [{"bbox": [1, 2, 3, 4], "is_checked": True, "page": 0}]}


def test_health_is_ok_after_startup() -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_detector_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="nope"), TestClient(create_app(Settings(detector="nope"))):
        pass
