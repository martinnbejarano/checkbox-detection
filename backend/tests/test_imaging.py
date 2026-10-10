from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from checkbox_detection.config import Settings
from checkbox_detection.imaging import load_pages
from checkbox_detection.imaging.errors import CorruptInput, InputTooLarge, UnsupportedFormat


def encode(image: Image.Image, fmt: str, **params: object) -> bytes:
    buf = BytesIO()
    image.save(buf, fmt, **params)
    return buf.getvalue()


def load_one(data: bytes, settings: Settings | None = None) -> np.ndarray:
    pages = load_pages(data, settings or Settings())
    assert [page.index for page in pages] == [0]
    return pages[0].image


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "TIFF", "WEBP"])
def test_decodes_allowed_formats_keeping_color(fmt: str) -> None:
    image = load_one(encode(Image.new("RGB", (40, 30), (200, 10, 10)), fmt))

    assert image.shape == (30, 40, 3)
    assert image.dtype == np.uint8


def test_grayscale_png_stays_2d() -> None:
    image = load_one(encode(Image.new("L", (40, 30), 128), "PNG"))

    assert image.shape == (30, 40)
    assert image[0, 0] == 128


def test_16_bit_gray_is_rescaled_to_8_bit() -> None:
    image = load_one(encode(Image.fromarray(np.full((30, 40), 40000, np.uint16)), "PNG"))

    assert image.shape == (30, 40)
    assert image[0, 0] == 40000 >> 8


def test_rgba_png_is_composited_on_white() -> None:
    rgba = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    rgba.paste((0, 0, 0, 255), (0, 0, 10, 10))

    image = load_one(encode(rgba, "PNG"))

    assert image.shape == (30, 40, 3)
    assert image[5, 5].tolist() == [0, 0, 0]
    assert image[20, 20].tolist() == [255, 255, 255]


def test_exif_rotated_jpeg_gives_upright_coordinates() -> None:
    # Stored 200 wide x 100 tall with a dark square at the stored top-left.
    # Orientation 6 means "rotate 90 degrees clockwise to display", which moves it to the top-right.
    stored = Image.new("L", (200, 100), 255)
    stored.paste(0, (0, 0, 20, 20))
    exif = Image.Exif()
    exif[0x0112] = 6

    image = load_one(encode(stored, "JPEG", exif=exif))

    assert image.shape == (200, 100)
    assert image[10, 90] < 50
    assert image[10, 10] > 200


def test_tiff_uses_first_frame() -> None:
    frames = [Image.new("L", (40, 30), 0), Image.new("L", (40, 30), 255)]

    image = load_one(encode(frames[0], "TIFF", save_all=True, append_images=frames[1:]))

    assert image[0, 0] == 0


def test_non_image_bytes_are_unsupported() -> None:
    with pytest.raises(UnsupportedFormat):
        load_pages(b"just some text", Settings())


def test_format_outside_allow_list_is_unsupported() -> None:
    with pytest.raises(UnsupportedFormat):
        load_pages(encode(Image.new("RGB", (40, 30)), "BMP"), Settings())


def test_truncated_image_is_corrupt() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (300, 300, 3), np.uint8)
    data = encode(Image.fromarray(noise), "PNG")

    with pytest.raises(CorruptInput):
        load_pages(data[: len(data) // 2], Settings())


def test_pixel_cap_is_checked_before_decoding() -> None:
    data = encode(Image.new("L", (40, 30)), "PNG")

    with pytest.raises(InputTooLarge):
        load_pages(data, Settings(max_image_pixels=40 * 30 - 1))
