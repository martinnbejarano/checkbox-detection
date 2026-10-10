import json
from collections.abc import Iterator

import cv2
import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from checkbox_detection.api.routes import get_pipeline
from checkbox_detection.config import Settings
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
    assert [b.keys() for b in boxes] == [{"bbox", "is_checked", "page"}] * 2
    assert [b["is_checked"] for b in boxes] == [False, True]
    assert [b["page"] for b in boxes] == [0, 0]
    for box, expected in zip(boxes, ([200, 400, 250, 450], [600, 400, 650, 450]), strict=True):
        assert np.allclose(box["bbox"], expected, atol=3)


def test_health_is_ok_after_startup() -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (Settings(locator="nope"), r"Unknown locator 'nope'; choose one of \['opencv'\]"),
        (
            Settings(classifier="nope"),
            r"Unknown classifier 'nope'; choose one of \['fill_ratio'\]",
        ),
    ],
    ids=["locator", "classifier"],
)
def test_unknown_stage_name_fails_at_startup(settings: Settings, message: str) -> None:
    with pytest.raises(ValueError, match=message), TestClient(create_app(settings)):
        pass


def assert_problem(response: httpx2.Response, status: int) -> None:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body.keys() == {"type", "title", "status", "detail"}
    assert body["status"] == status
    assert "Traceback" not in response.text


def test_declared_oversized_body_is_413() -> None:
    with TestClient(create_app(Settings(max_upload_bytes=1000))) as client:
        response = client.post("/detect", files={"file": ("page.png", b"x" * 2000)})

    assert_problem(response, 413)


def test_streamed_oversized_body_is_413() -> None:
    def chunks() -> Iterator[bytes]:
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="a"\r\n\r\n'
        yield b"x" * 2000
        yield b"\r\n--b--\r\n"

    with TestClient(create_app(Settings(max_upload_bytes=1000))) as client:
        response = client.post(
            "/detect",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=b"},
        )

    assert_problem(response, 413)


def test_unsupported_bytes_are_415() -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.post("/detect", files={"file": ("page.png", b"not an image")})

    assert_problem(response, 415)


def test_truncated_image_is_422() -> None:
    data = two_box_page()
    with TestClient(create_app(Settings())) as client:
        response = client.post("/detect", files={"file": ("page.png", data[: len(data) // 2])})

    assert_problem(response, 422)


def test_pixel_cap_exceeded_is_413() -> None:
    with TestClient(create_app(Settings(max_image_pixels=1000))) as client:
        response = client.post("/detect", files={"file": ("page.png", two_box_page())})

    assert_problem(response, 413)


def test_missing_file_field_is_422() -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.post("/detect", files={"other": ("page.png", b"x")})

    assert_problem(response, 422)


def test_unexpected_failure_is_500_without_stack_trace() -> None:
    class BrokenPipeline:
        def run(self, image: np.ndarray) -> list[object]:
            raise RuntimeError("secret internals")

    app = create_app(Settings())
    app.dependency_overrides[get_pipeline] = BrokenPipeline
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/detect", files={"file": ("page.png", two_box_page())})

    assert_problem(response, 500)
    assert "secret internals" not in response.text
    assert response.headers["x-request-id"]


def test_each_request_logs_one_json_line_with_timings(capsys: pytest.CaptureFixture[str]) -> None:
    with TestClient(create_app(Settings())) as client:
        response = client.post("/detect", files={"file": ("page.png", two_box_page())})

    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    [entry] = [line for line in lines if line.get("message") == "request"]
    assert entry["request_id"] == response.headers["x-request-id"]
    assert entry["status"] == 200
    assert entry["pages"] == 1
    assert entry["n_boxes"] == 2
    for key in ("total_ms", "decode_ms", "detect_ms"):
        assert entry[key] >= 0
