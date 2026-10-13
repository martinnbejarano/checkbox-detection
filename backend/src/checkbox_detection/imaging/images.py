"""Pillow decode: format allow-list, pixel cap, EXIF transpose, native gray or RGB."""

import warnings
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from checkbox_detection.imaging.errors import CorruptInput, InputTooLarge, UnsupportedFormat

_FORMATS = ["PNG", "JPEG", "TIFF", "WEBP"]

# Gray sources stay 2-D; expanding them to RGB triples memory and adds no information.
_GRAY_MODES = {"1", "L", "LA", "F"}

# Set once at import: toggling it per call would race between threadpool requests.
warnings.simplefilter("error", Image.DecompressionBombWarning)


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data), formats=_FORMATS)
    except UnidentifiedImageError as exc:
        raise UnsupportedFormat("Expected PNG, JPEG, TIFF or WEBP") from exc
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InputTooLarge(str(exc)) from exc

    # TODO: multi-page TIFF. Pillow opens frame 0, which is the only one decoded.
    with image:
        if image.width * image.height > max_pixels:
            raise InputTooLarge(
                f"Image is {image.width}x{image.height}, above the {max_pixels} pixel cap"
            )
        try:
            image.load()
            return _to_array(ImageOps.exif_transpose(image))
        except (OSError, SyntaxError, ValueError) as exc:
            raise CorruptInput(f"Could not decode the image: {exc}") from exc


def _to_array(image: Image.Image) -> np.ndarray:
    if image.mode.startswith("I"):
        # 16-bit gray; Pillow's convert("L") clips instead of rescaling.
        return np.asarray(np.clip(np.asarray(image), 0, 65535) >> 8, np.uint8)
    gray = image.mode in _GRAY_MODES
    if image.has_transparency_data:
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    return np.asarray(image.convert("L" if gray else "RGB"))
