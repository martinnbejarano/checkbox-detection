"""pypdfium2 rasterization: page and pixel caps, then each page to RGB under a global lock."""

import threading
from collections.abc import Iterator

import numpy as np
import pypdfium2 as pdfium

from checkbox_detection.imaging.errors import CorruptInput, InputTooLarge

# pdfium is not thread-safe, even across separate documents, so every call takes this lock.
# It is released between pages, so detection on a page runs outside it.
# Reentrant: a `render_pdf` generator dropped half-way closes its document from its `finally`,
# which can run on a thread already holding the lock.
# ponytail: global lock serializes PDF renders; scale throughput with worker processes.
_PDFIUM = threading.RLock()


def check_pdf(data: bytes, dpi: int, max_pages: int, max_pixels: int) -> int:
    """The page count, once the PDF opens and every page is within the caps. Renders nothing."""
    scale = dpi / 72  # PDF sizes are in points
    with _PDFIUM:
        pdf = _open(data)
        try:
            n_pages = len(pdf)
            if n_pages > max_pages:
                raise InputTooLarge(f"PDF has {n_pages} pages, above the {max_pages} page cap")
            for index in range(n_pages):
                width, height = pdf.get_page_size(index)
                pixels = width * scale * height * scale
                if pixels > max_pixels:
                    raise InputTooLarge(
                        f"Page {index} is {pixels:.0f} pixels at {dpi} DPI, "
                        f"above the {max_pixels} pixel cap"
                    )
            return n_pages
        except pdfium.PdfiumError as exc:
            raise CorruptInput(f"Could not read the PDF: {exc}") from exc
        finally:
            # Explicit close inside the lock: a finalizer would run unlocked on any thread.
            pdf.close()


def render_pdf(data: bytes, dpi: int) -> Iterator[np.ndarray]:
    """Each page of a PDF that passed `check_pdf`, rendered when asked for, so only one is held."""
    with _PDFIUM:
        pdf = _open(data)
        # Without this, checkboxes in fillable PDFs do not render.
        pdf.init_forms()
    try:
        for index in range(len(pdf)):
            with _PDFIUM:
                image = _render_page(pdf, index, dpi)
            yield image
    finally:
        with _PDFIUM:
            pdf.close()


def _open(data: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        if exc.err_code == pdfium.raw.FPDF_ERR_PASSWORD:
            raise CorruptInput("Encrypted PDFs are not supported") from exc
        raise CorruptInput(f"Could not read the PDF: {exc}") from exc


def _render_page(pdf: pdfium.PdfDocument, index: int, dpi: int) -> np.ndarray:
    page = pdf[index]
    try:
        # Always RGB: PDFs can carry colored watermarks and stamps.
        bitmap = page.render(scale=dpi / 72, rev_byteorder=True)
        image: np.ndarray = bitmap.to_numpy()
        bitmap.close()
        return image
    except pdfium.PdfiumError as exc:
        raise CorruptInput(f"Could not render page {index}: {exc}") from exc
    finally:
        page.close()
