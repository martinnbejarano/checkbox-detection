"""Turns uploaded bytes into validated, decoded pages (uint8 gray HxW or RGB HxWx3).

Raises `imaging.errors` exceptions.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from checkbox_detection.config import Settings
from checkbox_detection.imaging.images import decode_image
from checkbox_detection.imaging.pdf import check_pdf, render_pdf


@dataclass(frozen=True, slots=True)
class Page:
    index: int
    image: np.ndarray


def iter_pages(data: bytes, settings: Settings) -> tuple[int, Iterator[Page]]:
    """The page count and the pages. Every check runs here, before any page is returned; PDF
    pages are then rendered one at a time as the iterator is read, so only one is in memory."""
    # Readers accept the PDF header anywhere in the first 1024 bytes.
    if b"%PDF-" in data[:1024]:
        n_pages = check_pdf(
            data, settings.pdf_dpi, settings.max_pdf_pages, settings.max_image_pixels
        )
        images = render_pdf(data, settings.pdf_dpi)
        return n_pages, (Page(index, image) for index, image in enumerate(images))
    return 1, iter([Page(0, decode_image(data, settings.max_image_pixels))])


def load_pages(data: bytes, settings: Settings) -> list[Page]:
    """Every page at once, for scripts and tests."""
    return list(iter_pages(data, settings)[1])
