"""Turns uploaded bytes into validated, decoded pages (uint8 gray HxW or RGB HxWx3).

Raises `imaging.errors` exceptions. Must not import FastAPI, `api` or `detection`.
"""

from dataclasses import dataclass

import numpy as np

from checkbox_detection.config import Settings
from checkbox_detection.imaging.images import decode_image


@dataclass(frozen=True, slots=True)
class Page:
    index: int
    image: np.ndarray


def load_pages(data: bytes, settings: Settings) -> list[Page]:
    return [Page(0, decode_image(data, settings.max_image_pixels))]
