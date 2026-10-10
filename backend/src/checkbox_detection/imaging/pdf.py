"""pypdfium2 rasterization: page and pixel caps, each page rendered to RGB under a global lock."""

import threading

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from checkbox_detection.imaging.errors import CorruptInput, InputTooLarge

# pdfium is not thread-safe, even across separate documents, so every call takes this lock.
# ponytail: global lock serializes PDF renders; scale throughput with worker processes.
_PDFIUM = threading.Lock()


def render_pdf(data: bytes, dpi: int, max_pages: int, max_pixels: int) -> list[np.ndarray]:
    with _PDFIUM:
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            if exc.err_code == pdfium_c.FPDF_ERR_PASSWORD:
                raise CorruptInput("Encrypted PDFs are not supported") from exc
            raise CorruptInput(f"Could not read the PDF: {exc}") from exc
    try:
        with _PDFIUM:
            # Without this, checkboxes in fillable PDFs do not render.
            pdf.init_forms()
            n_pages = len(pdf)
        if n_pages > max_pages:
            raise InputTooLarge(f"PDF has {n_pages} pages, above the {max_pages} page cap")
        return [_render_page(pdf, index, dpi, max_pixels) for index in range(n_pages)]
    finally:
        # Explicit close inside the lock: a finalizer would run unlocked on any thread.
        with _PDFIUM:
            pdf.close()


def _render_page(pdf: pdfium.PdfDocument, index: int, dpi: int, max_pixels: int) -> np.ndarray:
    scale = dpi / 72  # PDF sizes are in points
    with _PDFIUM:
        page = pdf[index]
        try:
            width, height = page.get_size()
            pixels = width * scale * height * scale
            if pixels > max_pixels:
                raise InputTooLarge(
                    f"Page {index} is {pixels:.0f} pixels at {dpi} DPI, "
                    f"above the {max_pixels} pixel cap"
                )
            # Always RGB: PDFs can carry colored watermarks and stamps.
            bitmap = page.render(scale=scale, rev_byteorder=True)
            image: np.ndarray = bitmap.to_numpy()
            bitmap.close()
        except pdfium.PdfiumError as exc:
            raise CorruptInput(f"Could not render page {index}: {exc}") from exc
        finally:
            page.close()
    return image
