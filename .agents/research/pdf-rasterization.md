# PDF rasterization for `POST /detect` (research, 2026-10-07)

Question (issue #3): which library should turn uploaded PDFs into page images for
the OpenCV/ML checkbox detector? Constraints: Python 3.13, `python:3.13-slim`
(Debian trixie), FastAPI sync endpoints running in a threadpool, permissive license
(the service may be closed source). Versions are from `https://pypi.org/pypi/<name>/json` on 2026-10-07.

## Comparison

| | **pypdfium2** | pdf2image + poppler-utils | PyMuPDF | pdfplumber | Wand (ImageMagick + Ghostscript) |
|---|---|---|---|---|---|
| Latest | 5.14.0 (2026-10-04) | 1.17.0 (2024-01-07) | 1.28.2 (2026-08-06) | 0.11.10 (2026-06-15) | 0.7.2 (2026-06-25) |
| Engine | PDFium (Chrome's PDF engine) | poppler `pdftoppm` via subprocess | MuPDF | pdfminer.six for parsing, **pypdfium2 for rendering** | Ghostscript via ImageMagick |
| License | Apache-2.0 / BSD-3-Clause, PDFium BSD-style [S1] | MIT wrapper; poppler is GPL [S4] | **AGPL-3.0 or commercial** (Artifex) [S6] | MIT, plus pypdfium2 | MIT wrapper; **Ghostscript AGPL or commercial** [S9] |
| Wheels / system deps | `py3-none` ctypes wheels for manylinux_2_17 / musllinux (x86_64, aarch64, ...) and macOS 13+ arm64 / x86_64. Pure pip, no apt. | Pure-Python wheel, but needs `apt install poppler-utils` (and `brew install poppler` on dev Macs) [S4] | abi3 `cp310` wheels (cover 3.13) for linux and macOS. No apt. | Pure Python, pulls pdfminer.six, cryptography, pypdfium2 | Needs `libmagickwand` + `ghostscript` from apt |
| Footprint | 3.8 MB wheel (linux x86_64), about 8 MB installed (measured on macOS) | poppler-utils + libpoppler147 + cairo/nss/X11/fontconfig chain is about 10 MB of `.deb`s on trixie, more once unpacked [S5] | 25.8 MB wheel, about 58 MB installed (measured) | pdfminer.six + cryptography on top of pypdfium2 | Largest: ImageMagick plus Ghostscript (libgs10 alone is 2.5 MB compressed) |
| Speed, measured at 200 DPI grayscale, ms/page (scan PDF / vector PDF) | **36 / 3** | 89 / 9 (subprocess + PPM decode) | 49 / 2 | same as pypdfium2 plus overhead | not measured |
| DPI control | `render(scale=dpi/72)` [S2] | `dpi=`, `size=` | `get_pixmap(dpi=)` | `to_image(resolution=)` | `resolution=` |
| To numpy / PIL | `bitmap.to_numpy()` (zero-copy) and `.to_pil()`. `grayscale=True` gives a 2-D `uint8` array that goes straight into OpenCV. | PIL list (or file paths) | `pix.samples` → `np.frombuffer` | PIL | Wand image → blob |
| Memory | Render one page at a time. The bitmap buffer is Python-owned by default, so the array survives `bitmap.close()` [S3]. | Default returns **all pages in RAM**. README: "a relatively big PDF will use up all your memory" unless you pass `output_folder`/`paths_only` [S4]. | Page by page | Keeps pdfminer objects too | Heavy |
| Encrypted PDF | `PdfiumError` "Incorrect password", `err_code` set. Pass `password=` to open. | `PDFPageCountError` (parses stderr) | `ValueError` / `needs_pass` | via pypdfium2 | Ghostscript error |
| Truncated/corrupt (tested) | `PdfiumError` "Data format error" | `PDFPageCountError` | **"Repairs" silently** and rendered 20 pages from a 3 KB stub | via pdfminer | n/a |
| AcroForm checkboxes (tested) | **Invisible unless `pdf.init_forms()` is called first** (0 vs 444 dark px) [S2] | Drawn (464) | Drawn (468) | via pypdfium2, same caveat | n/a |
| Thread-safety | **Not thread-safe**, even across different documents. Serialize every pdfium call with a mutex, or use processes [S2]. | Separate process per call, so threads are safe; has `timeout=` | "Does not support running on multiple threads" [S7] | Inherits pdfium's limits | Process-level |
| Maintenance | Very active: release 3 days ago, repo pushed 2026-10-05 | Wrapper last commit 2024-01-07, no release since; poppler itself is active | Active | Active | Active |

Not considered: **pikepdf** (MPL-2.0, 10.16.0) and **pypdf** read and edit PDFs
but cannot render. **Calling `pdftoppm` directly** works, but it is still
poppler/GPL and a subprocess without the wrapper's error handling.

Benchmark setup: Apple M-series, Python 3.13.3, 20 pages. "scan" means the 4
images from `samples/` embedded in a PDF at 2550×4200 px per page. "vector" means
synthetic text plus rectangles at 1653×2339 px. Script: Pillow builds the PDF,
and each library renders it to a grayscale numpy array. These numbers are
indicative only and were not measured on linux/Docker.

## Recommendation: pypdfium2

1. **Use `pypdfium2==5.14.*`.** It has a permissive license, pip-only wheels
   (nothing to apt-install, and the dev Mac and the container behave the same),
   it was the fastest on scanned pages, gives direct grayscale numpy, and fails
   cleanly on corrupt or encrypted input. PyMuPDF is technically excellent, but the
   AGPL rules it out for a closed network service unless you buy a license.
   pdf2image adds a GPL system dependency and subprocess overhead, and the wrapper
   has had no release for 2.5 years. pdfplumber is pypdfium2 with a text-extraction
   layer on top that this service doesn't need.
2. **Thread-safety, which is required.** FastAPI runs `def` endpoints in a
   threadpool, and pdfium crashes or corrupts the process if two threads call it at
   the same time [S2]. Use one module-level `threading.Lock` and do every pdfium
   call under it: open, `len`, render, and **`close`**. Close explicitly, because a
   finalizer that runs later on another thread is an unlocked pdfium call. Hold the
   lock **per page**, not per request: render one page, copy nothing (the native
   buffer outlives `close`), release the lock, then run detection outside the lock.
   This also keeps memory at about one page. To get parallel PDF throughput, scale
   with uvicorn worker **processes** (each one has its own pdfium), not threads.
   ```python
   _PDFIUM = threading.Lock()  # ponytail: global lock; pdfium is process-global, scale with workers

   def iter_pdf_pages(data: bytes, dpi: int, max_pages: int, max_pixels: int):
       with _PDFIUM:
           pdf = pdfium.PdfDocument(data)          # PdfiumError -> 422 (corrupt) / 422 or 400 (password)
           pdf.init_forms()                        # must come before len()/pages, else form checkboxes vanish
           n = len(pdf)
       try:
           if n > max_pages: raise TooManyPages(n)  # -> 413
           for i in range(n):
               with _PDFIUM:
                   page = pdf[i]
                   w, h = page.get_size()            # PDF points; check before allocating
                   if (w * dpi / 72) * (h * dpi / 72) > max_pixels:
                       page.close(); raise PageTooLarge(i)  # -> 413
                   bmp = page.render(scale=dpi / 72, grayscale=True)
                   arr = bmp.to_numpy(); bmp.close(); page.close()
               yield i, arr                          # detection runs outside the lock
       finally:
           with _PDFIUM:
               pdf.close()
   ```
   Map `PdfiumError` to **422**. If `err_code == pdfium.raw.FPDF_ERR_PASSWORD`,
   return a clear message that password-protected PDFs aren't supported. Don't add a
   password parameter until someone asks for one.
3. **Default DPI: 200** for scanned forms. A 10 to 12 pt checkbox comes out at
   about 28 to 33 px, which is plenty for OpenCV contour or ML detection. Letter at
   200 DPI is 1700×2200, about 3.7 MP, or 3.7 MB as uint8 gray. 300 DPI is the usual
   OCR/scan standard, but it costs 2.25× the pixels and time. Make `dpi` a setting,
   not a request parameter, and **echo it in the response** together with the
   0-based `page` index of each box. Bbox coordinates are pixels of the rendered
   page at that DPI, so a client converts to PDF points with
   `pt = px * 72 / dpi`. The origin is top-left in the image, while PDF space is
   bottom-left; write that down in the README. If the detector's thresholds are
   tuned in pixels, the DPI must match what it was tuned on (`samples/` pages are
   2550 px wide, which is 300 DPI for Letter). So tune and render at the same DPI.
4. **Abuse limits** (settings, with 4xx problem+json):
   - Keep the upload byte cap (about 10 MiB) from `api-architecture.md` §5. It is
     the first guard.
   - `max_pdf_pages = 20` → 413. Check it right after `len(pdf)`, before rendering
     anything.
   - Per page, check `w_pt*h_pt*(dpi/72)^2 <= max_pixels` **before** `render()`.
     A PDF can declare a 200×200-inch page (14,400 pt is PDF's documented
     maximum), which would be 1.6 GP at 200 DPI. Reuse the same pixel budget as
     images (for example 50 MP).
   - Optionally add a total-pixel budget per request (pages × pixels) so 20
     maximum-size pages can't cost 1 GB.
   - pdfium render calls can't be interrupted from Python. The page and pixel caps
     are the real time bound. Put a request-level timeout at the proxy if needed.
5. **Detecting PDF vs image by magic bytes.** Don't trust `Content-Type` or the
   filename. PDFs start with `%PDF-` (ISO 32000 §7.5.2), but readers (Acrobat,
   pdfium) accept the header anywhere in the first 1024 bytes. Use
   `b"%PDF-" in data[:1024]` → PDF path. Everything else goes to Pillow with
   `formats=["PNG","JPEG","TIFF","WEBP"]`, which already sniffs PNG
   `89 50 4E 47 0D 0A 1A 0A`, JPEG `FF D8 FF`, TIFF `II*\0` / `MM\0*` and WEBP
   `RIFF....WEBP`, and returns 415 for anything else. No `python-magic`/libmagic is
   needed. Pdfium then validates the PDF properly: a fake `%PDF-` header surfaces
   as `PdfiumError` → 422.
6. **Install**: add `pypdfium2` to `pyproject.toml`. The Dockerfile needs no
   change, since there are no apt packages. This reverses the "reject PDF with 415"
   TODO in `api-architecture.md` §5.8, because PDF support is now in scope.

Skipped: password parameter, a multiprocessing render pool, and PDF text-layer
use (for example reading AcroForm checkbox *values* directly instead of
detecting them visually). Add them when a real input needs them. Reading form
field values would be exact for fillable PDFs, so it's a cheap follow-up.

## Sources

- [S1] pypdfium2 README and licensing: https://github.com/pypdfium2-team/pypdfium2#licensing ; PyPI: https://pypi.org/project/pypdfium2/
- [S2] pypdfium2 Python API, "Incompatibility with Threading", `PdfDocument`, `init_forms`, `PdfPage.render`: https://pypdfium2.readthedocs.io/en/stable/python_api.html
- [S3] pypdfium2 `PdfBitmap` source (native vs foreign buffers, `to_numpy` zero-copy): https://github.com/pypdfium2-team/pypdfium2/blob/main/src/pypdfium2/_helpers/bitmap.py
- [S4] pdf2image README and source (poppler requirement, `paths_only`, `timeout`, memory warning): https://github.com/Belval/pdf2image ; poppler license (GPL): https://poppler.freedesktop.org/
- [S5] Debian trixie `poppler-utils` 25.03.0 and dependencies: https://packages.debian.org/trixie/poppler-utils
- [S6] PyMuPDF licensing (AGPL / Artifex commercial): https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright
- [S7] PyMuPDF multiprocessing / threads: https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
- [S8] pdfplumber dependencies (pdfminer.six, pypdfium2): https://pypi.org/project/pdfplumber/
- [S9] Ghostscript licensing (AGPL / commercial): https://www.ghostscript.com/licensing/index.html
- [S10] PDF header: ISO 32000-1 §7.5.2 "File Header"; page-size limits are in Annex C (implementation limits).
- Local measurements (2026-10-07): scratch venv with pypdfium2 5.14.0, PyMuPDF 1.28.2, pdf2image 1.17.0 + poppler 25.05 (Homebrew). These covered render timing, encrypted/truncated handling and the AcroForm `init_forms` test. They were not committed.
