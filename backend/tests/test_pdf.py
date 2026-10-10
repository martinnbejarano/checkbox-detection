from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import cv2
import httpx2
import numpy as np
import pytest
from conftest import assert_problem
from fastapi.testclient import TestClient
from PIL import Image

from checkbox_detection.config import Settings
from checkbox_detection.main import create_app


def pdf_bytes(pages: list[np.ndarray], dpi: float) -> bytes:
    images = [Image.fromarray(page).convert("1") for page in pages]  # lossless, not JPEG
    buf = BytesIO()
    images[0].save(buf, "PDF", resolution=dpi, save_all=True, append_images=images[1:])
    return buf.getvalue()


def blank_page() -> np.ndarray:
    return np.full((1000, 1000), 255, np.uint8)


def two_page_pdf() -> bytes:
    """Drawn at 300 DPI, so pixels map 1:1. Empty box on page 0, X box elsewhere on page 1."""
    first, second = blank_page(), blank_page()
    cv2.rectangle(first, (200, 400), (250, 450), 0, 3)
    cv2.rectangle(second, (600, 100), (645, 145), 0, 3)
    cv2.line(second, (608, 108), (637, 137), 0, 4)
    cv2.line(second, (637, 108), (608, 137), 0, 4)
    return pdf_bytes([first, second], dpi=300)


def post(client: TestClient, data: bytes) -> httpx2.Response:
    return client.post("/detect", files={"file": ("doc.pdf", data)})


def test_two_page_pdf_returns_boxes_per_page_at_300_dpi() -> None:
    with TestClient(create_app(Settings())) as client:
        response = post(client, two_page_pdf())

    assert response.status_code == 200
    boxes = response.json()["boxes"]
    assert [(b["page"], b["is_checked"]) for b in boxes] == [(0, False), (1, True)]
    for box, expected in zip(boxes, ([200, 400, 250, 450], [600, 100, 645, 145]), strict=True):
        assert np.allclose(box["bbox"], expected, atol=5)  # pdfium anti-aliases edges


def test_pdf_above_page_cap_is_413() -> None:
    with TestClient(create_app(Settings(max_pdf_pages=1))) as client:
        response = post(client, two_page_pdf())

    assert_problem(response, 413)


def test_huge_pdf_page_is_413_before_rendering() -> None:
    # 100 px at 0.5 DPI is a 200-inch page: 60000 x 60000 px at 300 DPI, far too big to render.
    data = pdf_bytes([np.full((100, 100), 255, np.uint8)], dpi=0.5)

    with TestClient(create_app(Settings())) as client:
        response = post(client, data)

    assert_problem(response, 413)


def encrypted_pdf() -> bytes:
    data = pdf_bytes([blank_page()], dpi=300)
    encrypt = b"/Encrypt << /Filter /Standard /V 1 /R 2 /O <%s> /U <%s> /P -4 >> /ID [<00> <00>]"
    return data.replace(b"trailer\n<<", b"trailer\n<<" + encrypt % (b"00" * 32, b"00" * 32))


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        (encrypted_pdf(), "Encrypted PDFs are not supported"),
        (b"%PDF-1.4\nnot really a pdf", "Could not read the PDF"),
        (pdf_bytes([blank_page()], dpi=300)[:300], "Could not read the PDF"),
    ],
    ids=["encrypted", "garbage", "truncated"],
)
def test_unreadable_pdf_is_422(data: bytes, detail: str) -> None:
    with TestClient(create_app(Settings())) as client:
        response = post(client, data)

    assert_problem(response, 422)
    assert response.json()["detail"].startswith(detail)


def test_concurrent_pdf_requests_give_the_same_boxes() -> None:
    data = two_page_pdf()

    with TestClient(create_app(Settings())) as client, ThreadPoolExecutor(8) as pool:
        responses = list(pool.map(lambda _: post(client, data), range(64)))

    assert [r.status_code for r in responses] == [200] * 64
    assert all(r.json() == responses[0].json() for r in responses)
    assert len(responses[0].json()["boxes"]) == 2
