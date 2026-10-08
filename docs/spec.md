# Checkbox Detection Service: Implementation Spec

Status: ready to implement · 2026-10-08
Source of decisions: [Wayfinder map](https://github.com/martinnbejarano/checkbox-detection/issues/1). Each section links the ticket that holds the reasoning.
Background research: `docs/research/` (classical CV and services, deep learning and data, API architecture, PDF rasterization, public datasets).

---

## 1. Goal

Build an HTTP service with one endpoint, `POST /detect`. It finds every checkbox in a document image or PDF and says whether each one is checked. The inputs are US appraisal forms (Fannie Mae 1004, 1004MC and 1004C). Perfect accuracy is not required.

What makes the work good, in priority order:

1. **Readable code.** Someone new to the repo can tell where each step runs.
2. **Measured results.** Every accuracy claim comes from `scripts/evaluate.py`, not from looking at images.
3. **Works on unseen forms.** Nothing depends on the samples' box size or resolution (section 6, "Scale independence").
4. **Swappable detection.** Replacing the OpenCV detector with a learned model touches one module and one setting.
5. **Efficient and safe on CPU.** Uploads are validated before any expensive work.

## 2. API contract

### `POST /detect`

- **Request:** `multipart/form-data` with one field, `file`. Accepted formats: PNG, JPEG, TIFF (first frame only), WEBP and PDF. The format is detected from the file bytes. The `Content-Type` header and filename are ignored.
- **Response `200`:**

```json
{
  "boxes": [
    { "bbox": [x1, y1, x2, y2], "is_checked": true,  "page": 0 },
    { "bbox": [x1, y1, x2, y2], "is_checked": false, "page": 0 }
  ]
}
```

- `bbox`: integer pixel coordinates of the top-left and bottom-right corners, in the **upright** image (after EXIF rotation is applied). For PDFs, coordinates refer to the page rendered at 300 DPI.
- `page`: 0-based page index. It is always `0` for images. This field extends the challenge contract without breaking it: a client that ignores `page` still gets the exact challenge shape. ([multi-page ticket](https://github.com/martinnbejarano/checkbox-detection/issues/8))
- Box order: by page, then top to bottom, then left to right. A stable order makes the golden tests and diffs readable.
- If nothing is found, the response is `200` with `{"boxes": []}`. An empty result is not an error.

### `GET /health`

Returns `200 {"status": "ok"}` after startup has finished (detector loaded and warmed up). The Docker `HEALTHCHECK` uses it.

### Errors

Every error is RFC 9457 `application/problem+json` (`type`, `title`, `status`, `detail`). Errors never include stack traces.

| Case | Status |
|---|---|
| Body larger than `max_upload_bytes` (default 20 MiB) | 413 |
| PDF with more than `max_pdf_pages` pages (default 20) | 413 |
| Image or rendered page larger than `max_image_pixels` (default 50 MP) | 413 |
| Format not supported (detected from the bytes) | 415 |
| Corrupt or truncated file, or encrypted PDF | 422 |
| Missing `file` field | 422 (FastAPI default, mapped to problem+json) |
| Unexpected failure | 500 (logged with the request id) |

## 3. Architecture

([architecture ticket](https://github.com/martinnbejarano/checkbox-detection/issues/7))

Layered, with dependencies pointing one way: `api → imaging, detection`. `detection` imports nothing from the project except its own types.

```
HTTP request
   │
   ▼
api/routes.py ── reads bytes, enforces body limit
   │
   ▼
imaging/  ───── bytes → list[Page]      (sniff format, decode or rasterize, validate, EXIF, grayscale/RGB)
   │
   ▼
detection/ ──── Detector.detect(page) → list[Box]   (pure NumPy/OpenCV; no FastAPI, no I/O)
   │
   ▼
api/schemas.py ─ list[Box] → JSON response
```

### The one Protocol

`Detector` is the only abstract interface in the codebase. It exists because the [approach ticket](https://github.com/martinnbejarano/checkbox-detection/issues/4) requires that the detection method can be replaced (OpenCV now, a hybrid CNN or a fine-tuned detector later).

```python
class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]: ...
```

- Input: one page as an RGB `uint8` array (H×W×3). Each detector decides how to preprocess it. The OpenCV detector needs color to remove the red watermark and the blue cell fill.
- Output: boxes in that page's pixel coordinates. The API adds `page`.
- Implementation chosen at startup by `CHECKBOX_DETECTOR` (default `opencv`). A plain `dict` maps names to constructors in `detection/__init__.py`. There is no plugin registry.

There are no other ports, adapters, service layers, factories or DI containers. `imaging` and the PDF rasterizer are plain functions, because each has only one implementation.

### Domain types (`detection/types.py`)

```python
@dataclass(frozen=True, slots=True)
class Box:
    x1: int; y1: int; x2: int; y2: int
    is_checked: bool
    score: float  # detector confidence or fill ratio; internal, not in the response (used by evaluate.py)
```

## 4. Repository layout

```
checkbox-detection/
├── pyproject.toml            # uv project; ruff, mypy, pytest config
├── uv.lock
├── .python-version           # 3.13
├── Makefile                  # install, lint, typecheck, test, eval, run, docker-build, docker-run
├── Dockerfile
├── .dockerignore
├── README.md                 # the writeup: run, approach, results table, tradeoffs, limitations
├── docs/
│   ├── spec.md               # this file
│   └── research/             # research notes behind each decision
├── samples/                  # the 4 challenge images (sample-000 … sample-003.png)
├── frontend/                 # demo page: React + Vite + TS (section 13); built into frontend/dist
├── src/checkbox_detection/
│   ├── main.py               # create_app(): settings, logging, lifespan (load + warm up detector), routes, error handlers, static mount of frontend/dist
│   ├── config.py             # Settings (pydantic-settings, prefix CHECKBOX_)
│   ├── logging.py            # JSON log formatter + request-id/timing middleware
│   ├── api/
│   │   ├── routes.py         # POST /detect, GET /health; the only module with FastAPI routes
│   │   ├── schemas.py        # BoxOut, DetectResponse (Pydantic)
│   │   └── errors.py         # imaging/detection errors → problem+json
│   ├── imaging/
│   │   ├── __init__.py       # load_pages(data: bytes, settings) -> list[Page]
│   │   ├── errors.py         # UnsupportedFormat, InputTooLarge, CorruptInput (no HTTP knowledge)
│   │   ├── images.py         # Pillow decode: format allow-list, pixel cap, EXIF transpose, to RGB
│   │   └── pdf.py            # pypdfium2 rasterization under a global lock
│   └── detection/
│       ├── __init__.py       # Detector Protocol, DETECTORS = {"opencv": OpenCVDetector}, build_detector()
│       ├── types.py          # Box
│       ├── geometry.py       # iou(), nms(), sort_reading_order()
│       └── opencv/
│           ├── detector.py   # OpenCVDetector: wires the steps, holds parameters
│           ├── preprocess.py # color-overlay removal, binarization
│           ├── candidates.py # find_candidates(binary) -> list of rectangles
│           └── classify.py   # fill_ratio(), is_checked threshold
├── scripts/
│   ├── evaluate.py           # metrics against ground truth (section 7)
│   ├── prelabel.py           # runs the detector, writes ground-truth JSON drafts for correction
│   └── draw_boxes.py         # writes a PNG with boxes drawn, for visual review
└── tests/
    ├── conftest.py
    ├── data/ground_truth/    # sample-00N.json (hand-corrected labels)
    ├── test_geometry.py
    ├── test_opencv_detector.py  # synthetic images drawn with cv2
    ├── test_imaging.py
    ├── test_api.py
    └── test_golden.py        # real samples vs ground truth, with tolerances
```

Each package has a short module docstring that says what it does and what it must not import.

## 5. Image and PDF intake (`imaging/`)

Checks run in this order, cheapest first. Each failure raises an `imaging.errors` exception, and `api/errors.py` maps it to a status code.

1. **Body size.** Starlette `RequestBodyLimitMiddleware(max_body_size=settings.max_upload_bytes)` counts real bytes and returns 413. There is also a fallback `len(data)` check in the route.
2. **Format sniffing.**
   - If `b"%PDF-"` appears in the first 1024 bytes, the file goes to the PDF path.
   - Otherwise `Image.open(BytesIO(data), formats=["PNG", "JPEG", "TIFF", "WEBP"])` is used. `UnidentifiedImageError` means 415.
3. **Images.**
   - Check `width × height` against `max_image_pixels` before `load()`, and turn `DecompressionBombWarning` into an error.
   - Call `load()`. A failure here means 422.
   - Apply `ImageOps.exif_transpose`, convert to `RGB`, then `np.asarray`.
   - Multi-page TIFF: process frame 0 only. `TODO(prod): multi-page TIFF`.
4. **PDFs** ([PDF ticket](https://github.com/martinnbejarano/checkbox-detection/issues/3)). Use `pypdfium2`.
   - One module-level `threading.Lock` guards every pdfium call: open, `init_forms()`, `len`, render and close. pdfium is not thread-safe, even across separate documents.
   - Call `init_forms()` right after opening. Without it, checkboxes in fillable PDFs do not render.
   - Reject the file if the page count is above `max_pdf_pages` (413).
   - Before rendering a page, check its size in points × (300/72)² against `max_image_pixels` (413).
   - Render each page at `pdf_dpi` (default **300**) to RGB, release the lock, and return it.
   - Encrypted PDFs (`FPDF_ERR_PASSWORD`) and corrupt PDFs (`PdfiumError`) mean 422.
   - Close the document explicitly inside the lock. Do not rely on finalizers.

`Page` is `NamedTuple(index: int, image: np.ndarray)`.

`TODO(prod)`: read checkbox values straight from AcroForm fields when a PDF is fillable (exact answers, no vision).

## 6. Detection: the OpenCV detector

([approach ticket](https://github.com/martinnbejarano/checkbox-detection/issues/4), [priority ticket](https://github.com/martinnbejarano/checkbox-detection/issues/5))

### Facts measured on the samples

- Checkbox size **varies by form**. It is about 55×45 px on the 1004 sample (2550×4200), about 50×48 px on the 1004MC, and about 20×20 px on the 1004C sample (2550×3301). Boxes are not always square: the height-to-width ratio is about 0.8–1.0.
- Noise sources: table rules touching the boxes, blue cell fill (1004MC), a red diagonal watermark (1004C), handwritten strokes crossing the boxes (cropped sample), and text glyphs that look like boxes ("O", "D", "0").

### Scale independence (hard requirement)

The reviewers will test **other forms**, with other box sizes, resolutions and crops. The 4 samples are only examples. The detector must not encode their box size. ([scale ticket](https://github.com/martinnbejarano/checkbox-detection/issues/11))

Rules:

1. **Shape decides, size does not.** A candidate is accepted for being square-ish, closed, hollow, and having a thin border relative to its side. Size only rejects the extremes:
   - Minimum side `min_side_px` ≈ 8 px. Below that, the shape is noise and the state cannot be read.
   - Maximum side `max_side_frac` ≈ 6% of the page's short edge. Above that, the shape is a table cell, not a checkbox.
2. **Every kernel and window scales with the page.** These are all derived from the image size, never fixed pixels:
   - the adaptive-threshold block;
   - the line-removal kernel length, which is always longer than the largest allowed box side, so box edges survive;
   - the NMS distances.
3. **The page sets its own box size.** After shape filtering, compute the median side of the confident candidates. Then accept weaker candidates only if they are within ×0.5–×2 of that median. This uses the fact that a form usually has one or two box sizes. It removes square glyphs on forms with large boxes, without a hard-coded size. If there are fewer than 5 confident candidates, skip this step.
4. **Fill classification is relative.** The interior margin is a fraction of the measured side plus the measured border thickness. The ink ratio is a fraction of the interior area. Both work the same at 20 px and at 80 px.
5. **Very small inputs are upscaled.** If the page's short edge is below about 1000 px (a low-resolution photo or crop), upscale it ×2 before detection, then scale the boxes back to the original coordinates.

How this is verified:

- Synthetic tests draw boxes from 10 px to 100 px on the same page.
- `evaluate.py --scales 0.5,1,1.5` resizes every sample and runs the evaluation again. End-to-end accuracy must not drop by more than a few points across scales. The README reports this table.

### Pipeline

Each step is one small function, so a reader can follow the order in `detector.py`.

1. **`remove_color_overlays(rgb) -> gray`** (`preprocess.py`). Convert to HSV. Pixels with high saturation (red watermark, blue fill) become white, then convert to grayscale. Black ink has low saturation and survives.
2. **`binarize(gray) -> binary`**. Adaptive Gaussian threshold, inverted so ink is 255. The block size scales with page width.
3. **`find_candidates(binary) -> list[Rect]`** (`candidates.py`):
   - Remove long horizontal and vertical lines (morphological open with kernels much longer than a checkbox side), so boxes that touch table rules become separate. Keep the binary image from before this step for step 5.
   - `findContours(RETR_CCOMP)`. Keep a contour if all of these are true:
     - its side is between `min_side_px` and `max_side_frac` of the page (see "Scale independence");
     - its aspect ratio is between 0.7 and 1.4;
     - `approxPolyDP` gives 4 vertices, or the contour area divided by the bbox area is above 0.8;
     - it is hollow, meaning the ink ratio of the border ring is much higher than the interior's.
   - Do not require exactly one child contour, because an "X" splits the inner hole.
   - Apply the page-median size check (rule 3 of "Scale independence").
4. **`nms(rects, iou=0.3)`** (`geometry.py`). Merge duplicates, preferring the outer rectangle.
5. **`classify(binary_before_line_removal, rect) -> (is_checked, fill_ratio)`** (`classify.py`):
   - Shrink the rectangle inward by `max(border_thickness + 2px, 15% of side)`, then compute the ink ratio of the interior.
   - Threshold: Otsu over all fill ratios on the page when they split into two groups (the gap between groups is large enough). Otherwise use a fixed `fill_threshold` (default ≈ 0.10, tuned on the samples).
   - Ambiguous ratios (within `±margin` of the threshold) resolve to **`false`**, following the decision to avoid inventing checked boxes.
6. **`sort_reading_order(boxes)`**.

Every threshold lives in one frozen `OpenCVParams` dataclass with a comment per field. The defaults are tuned on the samples and are not read from the environment. Tuning favors **detection recall**: a missed box loses information, while an extra empty box is cheap to ignore downstream.

### Escalation path (built only if the numbers require it)

| Evidence from `evaluate.py` | Change | Scope |
|---|---|---|
| State accuracy is the main error (watermark, ✓, faint marks) | Replace `classify()` with a small ONNX CNN (`not_checkbox` / `unchecked` / `checked`) on 32×32 crops. Run it with `onnxruntime`. The API does not need PyTorch. | New `detection/hybrid/`. Reuses `opencv/candidates.py`. Registered as `"hybrid"`. |
| Detection recall is the main error (boxes fused with lines, hand-drawn strokes) | Fine-tune an Apache-2.0 detector (RF-DETR Nano/Small; YOLO26 only if AGPL is acceptable). Export to ONNX. Run on 1024 px tiles with global NMS. | New `detection/learned/`. Registered as `"learned"`. |

Training data for either path: synthetic pages rendered from the official blank 1004 / 1004MC / 1004C PDFs (every checkbox is a vector square, so labels are exact), plus real scans from Roboflow (861 public-domain housing forms, 2,830 CC BY pages). The 4 samples are never used for training. See [datasets ticket](https://github.com/martinnbejarano/checkbox-detection/issues/2) and `docs/research/public-checkbox-datasets.md`. Training code lives outside `src/` (`training/`) so the service never imports it.

## 7. Evaluation

([evaluation ticket](https://github.com/martinnbejarano/checkbox-detection/issues/6))

### Ground truth

`tests/data/ground_truth/sample-00N.json`:

```json
{
  "image": "samples/sample-000.png",
  "boxes": [
    { "bbox": [x1, y1, x2, y2], "is_checked": true, "tags": ["handwriting"] }
  ]
}
```

Allowed tags: `watermark`, `color_fill`, `handwriting`, `checkmark`, `partial_fill`, `touches_line`.

Labeling procedure:

1. Run `scripts/prelabel.py` to get a draft.
2. Correct the draft by hand with `scripts/draw_boxes.py` overlays, or in Label Studio if the draft is far off.
3. Commit the corrected JSON.

Budget: 1–2 hours.

### `scripts/evaluate.py`

- Matching: greedy by IoU, one-to-one, per page.
- Reports, per image and in total:
  - Detection precision, recall and F1 at IoU 0.5, plus recall at IoU 0.3.
  - State accuracy on matched boxes, with false-checked and false-unchecked counts shown separately.
  - **Headline: end-to-end accuracy**, where a ground-truth box counts only if it was matched at IoU ≥ 0.5 **and** its state is correct.
  - Results broken down by tag.
  - With `--scales`, the same metrics for each sample resized by each factor (scale-robustness table).
  - Latency p50 and p95 per page (runs the detector in-process; the Docker numbers are measured separately with `curl` timing).
- Output: a Markdown table pasted into the README.

## 8. Tests

([evaluation ticket](https://github.com/martinnbejarano/checkbox-detection/issues/6))

| File | What it proves |
|---|---|
| `test_geometry.py` | `iou`, `nms` and ordering on hand-made rectangles. |
| `test_opencv_detector.py` | On synthetic pages drawn with `cv2.rectangle`/`cv2.line`: empty, X, ✓ and filled boxes; boxes touching a table line; a red overlay; a page with only text, which gives `[]`. Box sides are parametrized from 10 px to 100 px, and page sizes from 1000 px to 5000 px wide. Asserts the count, bbox within ±3 px (or ±5% of the side), and `is_checked`. |
| `test_imaging.py` | PNG/JPEG/TIFF/WEBP decode; EXIF-rotated JPEG gives upright coordinates; truncated file gives `CorruptInput`; non-image bytes give `UnsupportedFormat`; pixel cap; a 2-page PDF made in-test gives 2 pages at 300 DPI; page cap; encrypted PDF. |
| `test_api.py` | `with TestClient(app)` so lifespan runs. Checks 200 schema and `page`, 413, 415, 422 problem+json, missing field, and `/health`. Uses `dependency_overrides` with a fake detector so HTTP behavior is tested on its own. |
| `test_golden.py` | Each real sample against its ground truth. Asserts end-to-end accuracy ≥ a floor recorded after tuning (for example 0.9), so a regression fails the test while small tweaks do not. |

## 9. Runtime and performance

- Endpoint declared with plain `def`. FastAPI runs it in its threadpool. OpenCV releases the GIL during heavy calls. pdfium calls are serialized by the lock.
- The detector is built once in `lifespan`, warmed up with one blank page, stored on `app.state`, and injected through a `get_detector` dependency.
- `cv2.setNumThreads(settings.cv_threads)` is set at startup (default 1 per worker), to avoid CPU oversubscription when there are several workers. Throughput is scaled with uvicorn workers or replicas, never with a process pool inside the app.
- Logged per request (JSON lines): request id, status, total ms, and also `pages`, `decode_ms`, `detect_ms` and `n_boxes`. The response carries an `X-Request-ID` header.
- Target: under 500 ms per 300 DPI page on 1 vCPU for the OpenCV detector. To be checked by evaluation; this is not a promise.

## 10. Configuration (`config.py`)

`Settings(BaseSettings)`, env prefix `CHECKBOX_`. The defaults run with no env vars.

| Setting | Default |
|---|---|
| `detector` | `"opencv"` |
| `max_upload_bytes` | 20 MiB |
| `max_image_pixels` | 50_000_000 |
| `max_pdf_pages` | 20 |
| `pdf_dpi` | 300 |
| `cv_threads` | 1 |
| `log_level` | `"INFO"` |

Detector tuning parameters are **not** settings. They live in code next to their explanation (section 6).

## 11. Tooling and code conventions

- **Python 3.13 and uv** (`uv.lock` committed). Runtime dependencies: `fastapi`, `uvicorn`, `pydantic-settings`, `python-multipart`, `opencv-python-headless`, `numpy`, `pillow`, `pypdfium2`. Dev dependencies: `pytest`, `httpx`, `ruff`, `mypy`.
- **ruff** for linting (rule sets E, F, I, B, UP, SIM, RUF) and formatting. **mypy --strict** on `src/`.
- **Make targets:** `install`, `lint`, `typecheck`, `test`, `eval`, `run`, `docker-build`, `docker-run`, `frontend-dev`, `frontend-build`. `make check` runs lint, typecheck and test.
- **Conventions:**
  - Functions are small and named for what they return (`find_candidates`, `fill_ratio`).
  - Full type hints everywhere.
  - No magic numbers outside `OpenCVParams` and `Settings`.
  - Each module docstring states its responsibility.
  - Comments explain *why*, not *what*.
  - `TODO(prod):` marks every known production gap, so `grep TODO(prod)` gives the list in the writeup.
- **Do not build:** OCR of checkbox labels (README next step only), job queues, a frontend beyond the demo page in section 13, process pools, a database or cache, API versioning, auth, a metrics stack, k8s manifests, image fetch by URL, extra Protocols or ABCs, or a config file format. Each of these appears as a `TODO(prod)` only where it would really go.

## 12. Packaging and deployment

([deployment ticket](https://github.com/martinnbejarano/checkbox-detection/issues/9))

- **Dockerfile:** multi-stage, following uv's official recipe.
  - Frontend stage: `node:22-slim`, which runs `npm ci && npm run build` in `frontend/` and produces `frontend/dist`.
  - Builder: `python:3.13-slim` plus a pinned `ghcr.io/astral-sh/uv`, running `uv sync --locked --no-dev`.
  - Runtime: `python:3.13-slim` with only `.venv`, `src` and the built `frontend/dist`. It runs as a non-root user with an exec-form `CMD ["uvicorn", "checkbox_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
  - `HEALTHCHECK` via python/urllib on `/health`.
  - No apt packages are needed: OpenCV is the headless wheel, and pypdfium2 ships its own binary.
- **Local (primary path):** `make docker-build && make docker-run`, or `make install && make run` without Docker.
- **Railway (demo URL):** the same image on 1 vCPU / 1–2 GB, with `PORT` taken from the environment. The README states the URL, the limits, and that it is a demo instance. `TODO(prod)`: auth and rate limiting in front of the public URL.

## 13. Demo frontend (`frontend/`)

([frontend ticket](https://github.com/martinnbejarano/checkbox-detection/issues/12))

A very small page for trying the API by hand. It is a testing aid, not a product. It must stay simple enough to read in two minutes.

- **Stack:** React, Vite and TypeScript, made from the official `react-ts` template. No UI library, no router and no state library. Plain CSS in one file. The browser's own `fetch` sends the request.
- **Behavior (one screen):**
  1. A drag-and-drop zone that also opens the file picker on click. It accepts images and PDFs.
  2. When a file is chosen, show a preview (images only) and the file name.
  3. A **Process** button that sends `POST /detect` with `FormData`. It is disabled while the request runs.
  4. The result:
     - The image, with each box drawn on top: green for checked, grey for unchecked. Coordinates are scaled from the image's natural size to its displayed size.
     - The raw response body as pretty-printed JSON, with a count of boxes found.
     - Errors (4xx/5xx) show the problem+json `title` and `detail`.
  5. For PDFs there is no overlay, only the JSON. `TODO`: render PDF pages in the browser for overlays.
- **Files:**
  - `frontend/src/App.tsx` holds the screen and its state: `file`, `status`, `result`, `error`.
  - `frontend/src/DropZone.tsx` handles file selection.
  - `frontend/src/BoxOverlay.tsx` draws the boxes.
  - `frontend/src/api.ts` has one function, `detect(file): Promise<DetectResponse>`, plus the response types. The types are written by hand to match `api/schemas.py`.
- **How it reaches the API:**
  - In development, `npm run dev` runs on Vite's port, and `vite.config.ts` proxies `/detect` to `http://localhost:8000`. No CORS setup is needed.
  - In Docker and on Railway, FastAPI serves the built `frontend/dist` with `StaticFiles` at `/`, mounted **after** the API routes so `/detect` and `/health` take priority. It is one container and one origin, so again no CORS. If `frontend/dist` does not exist (backend-only run), the mount is skipped.
- **Quality bar:** the `react-ts` template's ESLint config passes and `tsc` passes (`npm run build` runs both). No frontend unit tests. One manual check is listed in the README.
- **Make targets:** `make frontend-dev` and `make frontend-build`.

## 14. README (the writeup)

Sections:

1. Quick start (Docker, local, Railway URL, the demo page at `/`, and a `curl` example).
2. API contract (including the `page` extension and the error table).
3. Approach and why: classical first, measured, and swappable.
4. Results table from `evaluate.py`.
5. Alternatives considered and why they were not chosen: managed services (privacy, credentials, cost); VLMs (coarse coordinates on small boxes); fine-tuned detectors (data, CPU cost, AGPL); training from scratch.
6. Tradeoffs and known limitations. These include: thresholds tuned on 4 pages (mitigated by the scale-robustness table); box styles never seen in the samples (rounded corners, circles, dashed borders); the new URAR form replacing the 1004 family from 2026-11-02; first TIFF frame only; and the cap at 20 PDF pages.
7. Escalation path (section 6 table).
   Next step beyond the contract: extracting the text next to each box (the label, such as "Urban" or "Yes", and the question of its row). Not built: it needs an OCR dependency (for example RapidOCR, ONNX and pip-only), adds latency, and needs its own evaluation, and the challenge does not ask for it. ([label ticket](https://github.com/martinnbejarano/checkbox-detection/issues/13))
8. `TODO(prod)` list.

## 15. Build order

1. Scaffold: `pyproject.toml`, Makefile, `config.py`, `main.py`, and `/health` with a stub detector. Get `make check` green.
2. `imaging/` with its tests (images first, then PDF).
3. `detection/geometry.py` and the synthetic-image tests.
4. OpenCV detector, step by step. Use `draw_boxes.py` on each sample after each step.
5. `prelabel.py`, then hand-correct the ground truth, then `evaluate.py`, then tune `OpenCVParams` toward recall.
6. API errors, logging and golden tests.
7. Demo frontend, then the static mount in FastAPI.
8. Dockerfile (with the frontend stage), then the Railway deploy.
9. README with the measured results. Decide on escalation from the numbers.
