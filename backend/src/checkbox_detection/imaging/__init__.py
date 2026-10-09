"""Turns uploaded bytes into decoded pages (uint8 gray HxW or RGB HxWx3)."""

from dataclasses import dataclass

import numpy as np

from checkbox_detection.imaging.images import decode_image


@dataclass(frozen=True, slots=True)
class Page:
    index: int
    image: np.ndarray


def load_pages(data: bytes) -> list[Page]:
    # Format allow-list, size caps, typed errors, 16-bit rescaling and alpha-on-white
    # compositing arrive with upload validation (#16).
    return [Page(0, decode_image(data))]
