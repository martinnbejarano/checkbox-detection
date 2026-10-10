"""Turns uploaded bytes into validated, decoded pages (uint8 gray HxW or RGB HxWx3).

Raises `imaging.errors` exceptions. Must not import FastAPI, `api` or `detection`.
"""

from dataclasses import dataclass

import numpy as np

from checkbox_detection.config import Settings
from checkbox_detection.imaging.images import decode_image
from checkbox_detection.imaging.pdf import render_pdf


@dataclass(frozen=True, slots=True)
class Page:
    index: int
    image: np.ndarray


def load_pages(data: bytes, settings: Settings) -> list[Page]:
    # Readers accept the PDF header anywhere in the first 1024 bytes.
    # ponytail: every page is rendered before detection (~3 GB worst case at 20 x 50 MP RGB);
    # yield pages one at a time if memory becomes the limit.
    if b"%PDF-" in data[:1024]:
        images = render_pdf(
            data, settings.pdf_dpi, settings.max_pdf_pages, settings.max_image_pixels
        )
        return [Page(index, image) for index, image in enumerate(images)]
    return [Page(0, decode_image(data, settings.max_image_pixels))]
