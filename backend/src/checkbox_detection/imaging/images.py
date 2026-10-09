"""Pillow decode that keeps the source's gray or color mode without adding channels."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

# Gray sources stay 2-D; expanding them to RGB triples memory and adds no information.
_GRAY_MODES = {"1", "L", "LA", "I", "I;16"}


def decode_image(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        upright = ImageOps.exif_transpose(image)
        if upright.mode in _GRAY_MODES:
            return np.asarray(upright.convert("L"))
        return np.asarray(upright.convert("RGB"))
