# Checkbox Detection Service: Implementation Spec

Status: ready to implement · revised 2026-10-11 (tuning on public data; samples are eval only)
Source of decisions: [Wayfinder map](https://github.com/martinnbejarano/checkbox-detection/issues/1). Each section links the ticket that holds the reasoning.
Background research: `.agents/research/` (classical CV and services, deep learning and data, API architecture, PDF rasterization, public datasets, learned pipeline, pipeline code structure).

**Revision 2026-10-10.** Detection is now a 3-stage pipeline: locate → classify state → label (OCR). OpenCV stays as the baseline locator; a pretrained ONNX locator and state classifier are added next to it, and the evaluation picks the default. Label extraction reverses the earlier "not built" decision of [#13](https://github.com/martinnbejarano/checkbox-detection/issues/13): it is opt-in (`?labels=true`) so the challenge call does not pay for it. Model choices: `.agents/research/learned-pipeline.md`. Code layout: `.agents/research/pipeline-code-structure.md`.

**Revision 2026-10-11.** Four hand-labeled pages are too few to tune on. Every stage is now tuned on public datasets and synthetic pages (section 6, "Tuning on public data"): `yolox` and `cnn` get fine-tuned weights, and the `*Params` of every stage are chosen by a search on a validation split. The 4 samples, the external set and any page the user adds are **evaluation only**. This brings forward the fine-tuning that was planned for after the default-pipeline checkpoint.

---

## 1. Goal

Build an HTTP service with one endpoint, `POST /detect`. It finds every checkbox in a document image or PDF, says whether each one is checked and, on request, reads the label text next to it. The inputs are US appraisal forms (Fannie Mae 1004, 1004MC and 1004C). Perfect accuracy is not required.

What makes the work good, in priority order:

1. **Readable code.** Someone new to the repo can tell where each step runs.
2. **Measured results.** Every accuracy claim comes from `backend/scripts/evaluate.py`, not from looking at images.
3. **Works on unseen forms.** Nothing depends on the samples' box size or resolution (section 6, "Scale independence").
4. **Swappable stages.** Each pipeline stage (locate, classify) has more than one implementation, and switching one is one setting. Adding another touches one module and one dict entry.
5. **Efficient and safe on CPU.** Uploads are validated before any expensive work.

## 2. API contract

### `POST /detect`

- **Request:** `multipart/form-data` with one field, `file`. Accepted formats: PNG, JPEG, TIFF (first frame only), WEBP and PDF. The format is detected from the file bytes. The `Content-Type` header and filename are ignored.
- **Query parameter `labels`** (bool, default `false`): when `true`, each box also carries `label`.
- **Response `200`:**

```json
{
  "boxes": [
    { "bbox": [x1, y1, x2, y2], "is_checked": true,  "page": 0, "score": 0.91 },
    { "bbox": [x1, y1, x2, y2], "is_checked": false, "page": 0, "score": 0.34 }
  ]
}
```

With `?labels=true`:

```json
{ "boxes": [ { "bbox": [x1, y1, x2, y2], "is_checked": true, "page": 0, "score": 0.91, "label": "Urban" } ] }
```

- `bbox`: integer pixel coordinates of the top-left and bottom-right corners, in the **upright** image (after EXIF rotation is applied). For PDFs, coordinates refer to the page rendered at 300 DPI.
- `page`: 0-based page index. It is always `0` for images. This field extends the challenge contract without breaking it: a client that ignores `page` still gets the exact challenge shape. ([multi-page ticket](https://github.com/martinnbejarano/checkbox-detection/issues/8))
- `score`: the locator's confidence that the box is a checkbox, in [0, 1] (stage 1, `Rect.score`). YOLOX gives objectness × class probability; the OpenCV locator has no confidence and always returns `1.0`. The demo page flags low-confidence boxes with it. Like `page`, it is an extra field: a client that ignores it still gets the challenge shape.
- `label`: present only when `labels=true`. The text OCR'd from the strip right of the box (section 6, stage 3), or `null` when nothing is read. Like `page`, it extends the contract without breaking it.
- Box order: by page, then top to bottom, then left to right. A stable order makes the golden tests and diffs readable.
- If nothing is found, the response is `200` with `{"boxes": []}`. An empty result is not an error.

### `GET /health`

Returns `200 {"status": "ok"}` after startup has finished (pipeline built, selected models loaded). The Docker `HEALTHCHECK` uses it.

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

([architecture ticket](https://github.com/martinnbejarano/checkbox-detection/issues/7); layout patterns in `.agents/research/pipeline-code-structure.md`)

Layered, with dependencies pointing one way: `api → imaging, detection`. `detection` imports nothing from the project except its own modules. It does not import `config`: `main.py` passes plain values in.

```
HTTP request
   │
   ▼
api/routes.py ── reads bytes, enforces body limit, reads ?labels
   │
   ▼
imaging/  ───── bytes → list[Page]      (sniff format, decode or rasterize, validate, EXIF, keep native gray/color)
   │
   ▼
detection/ ──── Pipeline.run(page) → list[Box], then Pipeline.label(page, boxes) if labels   (pure NumPy/OpenCV/onnxruntime; no FastAPI, no I/O)
   │              1. Locator.locate(page)              → list[Rect]          where are the boxes
   │              2. StateClassifier.classify(page, …) → list[(bool, score)] is each one checked
   │              3. RapidOcrLabeler.label(page, …)    → list[str | None]   what does it say (only if labels)
   ▼
api/schemas.py ─ list[Box] → JSON response
```

### The pipeline and its two Protocols

One object per stage that owns its model and its pre/post-processing, plus one thin composer that calls the stages in order. 

```python
class Locator(Protocol):
    def locate(self, image: np.ndarray) -> list[Rect]: ...

class StateClassifier(Protocol):
    # Batched: all rects of a page in one call, so an ONNX model does one session.run.
    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]: ...

@dataclass(frozen=True)
class Pipeline:
    locator: Locator
    classifier: StateClassifier
    labeler: RapidOcrLabeler

    def run(self, image: np.ndarray) -> list[Box]:
        rects = self.locator.locate(image)
        states = self.classifier.classify(image, rects)
        return sort_reading_order([Box(r, c, s) for r, (c, s) in zip(rects, states, strict=True)])

    def label(self, image: np.ndarray, boxes: list[Box]) -> list[Box]:
        texts = self.labeler.label(image, [b.rect for b in boxes])
        return [replace(b, label=t) for b, t in zip(boxes, texts, strict=True)]
```

- `label` is a separate method so the route can skip it and log its time as `label_ms`.

- `run()` makes three named calls, so the stage order is visible in one place. A generic list of stages (docling's `build_pipe`) would hide it.
- `Locator` and `StateClassifier` are `typing.Protocol`s because each has two implementations chosen by config. Test fakes need no inheritance.
- The labeler has one implementation, so it is a concrete class with no Protocol.
- Input: one page as a `uint8` array, **grayscale (H×W)** or **RGB (H×W×3)**, whichever the source had. Each stage decides how to preprocess it.
- Output: boxes in that page's pixel coordinates. The API adds `page`.
- Implementations are chosen at startup by `CHECKBOX_LOCATOR` and `CHECKBOX_CLASSIFIER`. Two plain dicts in `detection/__init__.py` map names to constructors; an unknown name raises `ValueError` listing the choices. `scripts/evaluate.py` loops over the same dicts to compare combinations. There is no plugin registry.

There are no other ports, adapters, service layers, factories or DI containers. `imaging` and the PDF rasterizer are plain functions, because each has only one implementation.

### Domain types (`detection/types.py`)

```python
@dataclass(frozen=True, slots=True)
class Rect:  # output of stage 1, input of stages 2 and 3
    x1: int; y1: int; x2: int; y2: int
    score: float  # locator confidence; 1.0 for OpenCV, which has none

@dataclass(frozen=True, slots=True)
class Box:  # what the API serializes
    rect: Rect
    is_checked: bool
    checked_score: float  # P(checked) for the CNN, interior ink ratio for fill_ratio; internal (used by evaluate.py)
    label: str | None = None  # filled only when labels were requested
```

## 4. Repository layout

```
checkbox-detection/
├── README.md                 # the writeup: run, approach, results table, tradeoffs, limitations
├── Makefile                  # single entry point; delegates to backend/ (uv) and frontend/ (npm)
├── railway.toml              # points Railway at backend/Dockerfile (build context = repo root)
├── .github/workflows/check.yml  # runs `make check` on every PR
├── .claude/skills/testing/   # how tests are planned and written (section 8)
├── .agents/                  # agent working material, not user docs: this spec + research notes behind each decision
│   ├── spec.md               # this file
│   └── research/
├── samples/                  # the 4 challenge images; shared input data for backend tests/eval and manual frontend testing
├── frontend/                 # demo page: React + Vite + TS (section 13); built into frontend/dist
└── backend/                  # everything Python: service, scripts, tests, container
    ├── pyproject.toml        # uv project; ruff, mypy, pytest config
    ├── uv.lock
    ├── .python-version       # 3.13
    ├── models.txt            # pinned model files: HF repo, file, full commit sha (read by `make models`)
    ├── models/               # pretrained .onnx from `make models` (gitignored); fine-tuned *-tuned.onnx (committed)
    ├── Dockerfile            # multi-stage; builds frontend and copies the tuned models, so build context is the repo root
    ├── Dockerfile.dockerignore  # BuildKit picks this up for backend/Dockerfile
    ├── src/checkbox_detection/
    │   ├── main.py           # create_app(): settings, logging, lifespan (build pipeline), routes, error handlers, optional static mount
    │   ├── config.py         # Settings (pydantic-settings, prefix CHECKBOX_)
    │   ├── logging.py        # JSON log formatter + request-id/timing middleware
    │   ├── api/
    │   │   ├── routes.py     # POST /detect, GET /health; the only module with FastAPI routes
    │   │   ├── schemas.py    # BoxOut, DetectResponse (Pydantic)
    │   │   └── errors.py     # imaging/detection errors → problem+json
    │   ├── imaging/
    │   │   ├── __init__.py   # load_pages(data: bytes, settings) -> list[Page]
    │   │   ├── errors.py     # UnsupportedFormat, InputTooLarge, CorruptInput (no HTTP knowledge)
    │   │   ├── images.py     # Pillow decode: format allow-list, pixel cap, EXIF transpose, keep L or RGB
    │   │   └── pdf.py        # pypdfium2 rasterization under a global lock
    │   └── detection/
    │       ├── __init__.py   # Pipeline, LOCATORS, CLASSIFIERS, build_pipeline()  ← the file to read first
    │       ├── types.py      # Rect, Box
    │       ├── geometry.py   # iou(), nms(), sort_reading_order(), tile_windows()
    │       ├── binarize.py   # to_clean_gray(), binarize(); shared by the OpenCV locator and fill_ratio
    │       ├── onnx.py       # load_session(path, threads): fails fast if the file is missing
    │       ├── locate/       # stage 1
    │       │   ├── __init__.py   # Locator Protocol
    │       │   ├── opencv.py     # OpenCVLocator + OpenCVParams (baseline)
    │       │   └── yolox.py      # YoloxLocator + YoloxParams: tile → letterbox → run → decode → global NMS
    │       ├── classify/     # stage 2
    │       │   ├── __init__.py   # StateClassifier Protocol
    │       │   ├── fill_ratio.py # FillRatioClassifier + FillParams
    │       │   └── state_cnn.py  # StateCnnClassifier: crop → 40×160 gray letterbox → one batched run
    │       └── label.py      # stage 3: RapidOcrLabeler (label strip per box → RapidOCR recognition only)
    ├── training/             # offline tuning (section 6), own uv project with PyTorch; never imported by src/
    │   ├── data.py           # Roboflow scans + synthetic 1004 pages → COCO train/val split under data/ (not committed)
    │   ├── train_yolox.py    # fine-tune form-field-v1-nano, export ONNX, sweep YoloxParams
    │   ├── train_cnn.py      # fine-tune form-field-v1-state, export ONNX, sweep StateCnnParams
    │   └── search.py         # OpenCVParams, BinarizeParams, FillParams by search on validation
    ├── scripts/
    │   ├── evaluate.py       # metrics against ground truth, per locator × classifier combo (section 7)
    │   ├── prelabel.py       # runs the pipeline, writes ground-truth JSON drafts for correction
    │   └── draw_boxes.py     # writes a PNG with boxes drawn, for visual review
    └── tests/                # strategy in section 8; rules in .claude/skills/testing/
        ├── conftest.py       # samples_dir, models_dir, png_bytes(), blank_page()
        ├── data/ground_truth/    # sample-00N.json (labels for samples/)
        ├── data/external/        # unseen external evaluation pages + labels
        ├── test_api.py       # integration: contract, errors, labels, logging
        ├── test_pdf.py       # integration: multi-page PDFs through /detect
        ├── test_golden.py    # integration: each sample × each combo ≥ recorded floor
        ├── test_locate.py    # unit: scale grid, gray vs RGB, table lines, tile edges
        ├── test_classify.py  # unit: one case per mark class, ambiguous → unchecked
        ├── test_label.py     # unit: strip geometry + one real OCR read
        ├── test_geometry.py  # pure
        ├── test_imaging.py   # decode rules
        └── test_evaluate.py  # pure: matching and label normalization
```

Where each stage's code lives: pre- and post-processing stay in the module that owns the model (letterbox, tiling and YOLOX decode in `locate/yolox.py`; crop and resize in `classify/state_cnn.py`), as in OnnxTR, without separate `PreProcessor`/`PostProcessor` classes. `onnx.py` is the only place that builds an `InferenceSession`, so the thread setting and the "model file missing" error are written once.

**Rule:** `frontend/` and `backend/` are independent projects that share only the HTTP contract. Neither imports from the other. The root holds only what spans both: the README, the Makefile, the Railway config, the shared samples and the agent material.

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
   - Apply `ImageOps.exif_transpose`.
   - Normalize the mode **without adding channels**:
     - `L` and `1` become `L`;
     - `LA`, `I;16` and `I` become `L` (8-bit);
     - `RGB`, `RGBA`, `P` and `CMYK` become `RGB`, with alpha composited on white.
   - Then call `np.asarray`.
   - A grayscale scan is never expanded to RGB. Expanding would triple its memory and add no information.
   - Color is kept when the source has it, because color is how overlays are told apart from ink. Converting RGB to gray loses that information and cannot be undone.
   - Multi-page TIFF: process frame 0 only. `TODO(prod): multi-page TIFF`.
4. **PDFs** ([PDF ticket](https://github.com/martinnbejarano/checkbox-detection/issues/3)). Use `pypdfium2`.
   - One module-level `threading.Lock` guards every pdfium call: open, `init_forms()`, `len`, render and close. pdfium is not thread-safe, even across separate documents.
   - Call `init_forms()` right after opening. Without it, checkboxes in fillable PDFs do not render.
   - Reject the file if the page count is above `max_pdf_pages` (413).
   - Before rendering a page, check its size in points × (300/72)² against `max_image_pixels` (413).
   - Render each page at `pdf_dpi` (default **300**) to RGB, release the lock, and return it. The render is always RGB, because PDFs can carry colored watermarks and stamps.
   - Encrypted PDFs (`FPDF_ERR_PASSWORD`) and corrupt PDFs (`PdfiumError`) mean 422.
   - Close the document explicitly inside the lock. Do not rely on finalizers.

`Page` is `NamedTuple(index: int, image: np.ndarray)`.

`TODO(prod)`: read checkbox values straight from AcroForm fields when a PDF is fillable (exact answers, no vision).

## 6. Detection pipeline

([approach ticket](https://github.com/martinnbejarano/checkbox-detection/issues/4), [priority ticket](https://github.com/martinnbejarano/checkbox-detection/issues/5); model choices and CPU probes in `.agents/research/learned-pipeline.md`)

| Stage | Implementations (config name) | Module |
|---|---|---|
| 1. Locate | `opencv` (baseline, no weights) · `yolox` (`nutrientdocs/form-field-v1-nano`, Apache-2.0, fine-tuned) | `locate/opencv.py`, `locate/yolox.py` |
| 2. Classify state | `fill_ratio` (no weights) · `cnn` (`nutrientdocs/form-field-v1-state`, Apache-2.0, fine-tuned) | `classify/fill_ratio.py`, `classify/state_cnn.py` |
| 3. Label | RapidOCR recognition only, run only when `labels=true` | `label.py` |

Any locator works with any classifier. The default pair is chosen at the [default-pipeline checkpoint](https://github.com/martinnbejarano/checkbox-detection/issues/22) from the evaluation tables: `yolox` + `cnn`, both fine-tuned (`.agents/research/tuning-results.md`). `opencv` + `fill_ratio` stays as the baseline that needs no model files.

Rejected after a zero-shot probe on the samples (details in the research note): Docling layout (640 px squashed input, box covers checkbox plus label, state often wrong), FFDNet S/L (AGPL, misses checked boxes, L finds nothing on 3 of 4 samples), `wendys-llc/checkbox-detector` (AGPL in practice, low checked recall). The README lists them under alternatives.

### Facts measured on the samples

- Checkbox size **varies by form**. It is about 55×45 px on the 1004 sample (2550×4200), about 50×48 px on the 1004MC, and about 20×20 px on the 1004C sample (2550×3301). Boxes are not always square: the height-to-width ratio is about 0.8–1.0.
- Noise sources: table rules touching the boxes, blue cell fill (1004MC), a red diagonal watermark (1004C), handwritten strokes crossing the boxes (cropped sample), and text glyphs that look like boxes ("O", "D", "0").

### Scale independence (hard requirement)

This system will test **other forms**, with other box sizes, resolutions and crops. The 4 samples are only examples. No stage may encode their box size. ([scale ticket](https://github.com/martinnbejarano/checkbox-detection/issues/11))

Rules for the OpenCV locator and `fill_ratio`:

1. **Shape decides, size does not.** A candidate is accepted for being square-ish, closed, hollow, and having a thin border relative to its side. Size only rejects the extremes:
   - Minimum side `min_side_px` ≈ 8 px. Below that, the shape is noise and the state cannot be read.
   - Maximum side `max_side_frac` ≈ 6% of the page's short edge. Above that, the shape is a table cell, not a checkbox.
2. **Every kernel and window scales with the page.** These are all derived from the image size, never fixed pixels:
   - the adaptive-threshold block;
   - the line-removal kernel length, which is always longer than the largest allowed box side, so box edges survive;
   - the NMS distances.
3. **The page sets its own box size.** After shape filtering, compute the median side of the confident candidates (4 vertices, rectangularity ≥ 0.86, border ≤ 11% of the side). Then accept weaker candidates only if they are within ×0.75–×1.33 of that median. This uses the fact that a form usually has one or two box sizes. It removes square glyphs, without a hard-coded size: on the samples, real weak boxes sit at 0.97–1.06 of the median and letters at 0.45–0.65. If there are fewer than 5 confident candidates, weak candidates are dropped, so a text-only page returns no boxes.
4. **Fill classification is relative.** The interior margin is a fraction of the measured side plus the measured border thickness. The ink ratio is a fraction of the interior area. Both work the same at 20 px and at 80 px.
5. **Very small inputs are upscaled.** If the page's short edge is below about 1000 px (a low-resolution photo or crop), upscale it ×2 (nearest neighbour, so thin borders stay thin) before detection, then scale the boxes back to the original coordinates.

Rule for the YOLOX locator: the model sees fixed 640 px inputs, so the page is cut into tiles whose size is a fraction of the page **width** (`tile_frac` 0.6, ≈ 1530 px on a 2550 px page), not a fixed pixel count. Width rather than the short edge, so a wide crop (the 1586×846 sample) is not cut into tiles that blow its boxes up. The pretrained model was sensitive to the tile size (validation F2 0.52–0.89 over `tile_frac` 0.5–1.0); the fine-tuned one is not (0.930–0.946), so one tile size covers the box sizes seen so far.

How this is verified:

- Synthetic tests draw boxes from 10 px to 100 px on the same page.
- `evaluate.py --scales 0.5,1,1.5` resizes every sample and runs the evaluation again, for every combo. End-to-end accuracy must not drop by more than a few points across scales. The README reports this table.

### Stage 1a: OpenCV locator (`locate/opencv.py`)

Each step is one small function, so a reader can follow the order in `OpenCVLocator.locate`.

1. **`to_clean_gray(image) -> gray`** (`binarize.py`).
   - If the image is RGB: convert to HSV, turn pixels with high saturation (red watermark, blue fill) white, then convert to grayscale. Black ink has low saturation and survives.
   - If the image is already grayscale: return it unchanged.
   - Cost: about 3× the memory of a gray page while this step runs (≈32 MB vs ≈11 MB at 2550×4200) and a few milliseconds of time. This is small next to the false positives it prevents.
2. **`binarize(gray) -> binary`** (`binarize.py`). Adaptive Gaussian threshold, inverted so ink is 255. The block size scales with page width.
3. **`find_candidates(binary) -> list[Rect]`**:
   - Remove long horizontal and vertical lines (morphological open with kernels much longer than a checkbox side), so boxes that touch table rules become separate.
   - `findContours(RETR_CCOMP)`. Keep a contour if all of these are true:
     - its side is between `min_side_px` and `max_side_frac` of the page (see "Scale independence");
     - its aspect ratio is between 0.7 and 1.4;
     - `approxPolyDP` gives 4 vertices, or the contour area divided by the bbox area is above 0.8;
     - it is hollow, meaning its holes cover at least 25% of its bbox. An X or a fill with a gap still leaves holes; a solid blob does not.
   - Do not require exactly one child contour, because an "X" splits the inner hole.
   - Apply the page-median size check (rule 3 of "Scale independence").
4. **`nms(rects, iou=0.3)`** (`geometry.py`). Merge duplicates, preferring the outer rectangle.

Every threshold lives in one frozen `OpenCVParams` dataclass with a comment per field. Tuning favors **detection recall**: a missed box loses information, while an extra empty box is cheap to ignore downstream.

### Stage 1b: YOLOX locator (`locate/yolox.py`)

`nutrientdocs/form-field-v1-nano`: YOLOX-Nano, 0.9M params, 3.7 MB ONNX, classes Text / ChoiceButton / Signature. Only `ChoiceButton` is kept.

1. **`tile_windows(width, height, tile, overlap) -> list[window]`** (`geometry.py`): overlapping tiles covering the page (overlap larger than the largest allowed box side, so no box is cut in every tile). The page goes through `to_clean_gray` first (red watermark and blue fill removed), then to 3 channels.
2. Per tile: letterbox to 640×640, one `session.run`, decode to page coordinates, keep `ChoiceButton` above `score_threshold`.
3. **`nms`** across all tiles (global), then drop rects outside the size extremes of rule 1 (the upper one as a fraction of the page width).

Probe on one M4 thread: 50–270 ms per page. Parameters (tile size, overlap, score threshold 0.2 for recall, NMS IoU) live in a frozen `YoloxParams`.

### Stage 2a: `fill_ratio` classifier (`classify/fill_ratio.py`)

- Binarize the page with `binarize.py` **before** line removal.
- Per rect: shrink inward by `max(border_thickness + 5% of side, 20% of side)`, then compute the ink ratio of the interior.
- Only ink from strokes that sit mostly inside the box counts (labeling policy, §7). In a crop one side wider than the box, the box edges (straight runs ≥ 80% of the side) are set apart, and each remaining stroke counts only if ≥ 50% of its ink is inside the box. A stroke takes in the edge ink within about one border of it, so a stray stroke that crosses the edge stays one stroke, mostly outside the box.
- Threshold: a fixed `fill_threshold` (0.07: 0.05 plus a 0.02 margin, so ambiguous ratios resolve to **`false`**, following the decision to avoid inventing checked boxes). It is kept against validation, where 0.02 scores 1.6 points higher: a 2% speck would then read as checked. A per-page split (threshold in the widest gap between the ratios) was tried and dropped: on the cropped sample it split faint X marks from full ones and called the X marks empty.
- Parameters in a frozen `FillParams`.

### Stage 2b: CNN state classifier (`classify/state_cnn.py`)

`nutrientdocs/form-field-v1-state`, fine-tuned: a 96k-param CNN on 40×160 gray crops that outputs empty / filled. Probe: about 0.2 ms per crop.

- Crop each rect with padding, convert to clean gray (`binarize.to_clean_gray`), letterbox to 40×160, and stack all crops of the page into **one batched** `session.run`.
- `checked_score` = P(filled). `threshold` (`StateCnnParams`) is 0.7: the model's 0.5 plus a 0.2 margin, so an ambiguous score resolves to **`false`**, as for `fill_ratio`. A third "ambiguous" class is not trained.
- Known limit: no stray-stroke handling; a stroke crossing an empty box reads as checked. On the external set the fine-tuned model marks no empty box checked, but misses large pen strokes drawn mostly outside a small box (35 of 419 boxes), which the third party labelled checked and the labeling policy does not.

### Stage 3: label (`label.py`)

`RapidOcrLabeler`, RapidOCR 3.x (Apache-2.0, ONNX, PP-OCR models bundled in the wheel), **recognition only** (`use_det=False`, `use_cls=False`). Probe: about 9 ms per crop. Full-page OCR is not used: it took about 10 s and 2.5 GB per page.

- Runs only when the request has `labels=true`.
- For each box, crop a strip of the clean gray page (`binarize.to_clean_gray`) to its right: same row band (box height plus 25% of the side above and below), from the box's right edge to the next box on the same row, 15 box sides, or the page edge.
- Cut the strip where the label ends: at the first vertical rule (a column inked over ≥ 90% of the band) or after a blank run of one box side following the text. Underlines and row rules are erased first (`binarize.remove_lines`) so they do not count as text. A strip with no text is `None` without running OCR.
- Pad the strip with white left and right (the recognizer drops a letter touching the crop edge), recognize it, strip whitespace; empty text or confidence below 0.5 becomes `None`.
- Parameters in a frozen `LabelParams`.
- Known limits on the samples: long sentence labels ("did not analyze the contract…") come out garbled, and boxes whose label is on their left (the utilities table) read `None` or a stray glyph.
- Built and warmed up once in `lifespan` on a blank strip, because RapidOCR loads models lazily on first call. Its thread setting is forced to `onnx_threads` (its default is all cores).
- `TODO(prod)`: labels to the left of the box, and the question of the row.

### Tuning on public data ([tuning ticket](https://github.com/martinnbejarano/checkbox-detection/issues/21))

No public dataset has bbox + state + label together, so each stage uses the sources that carry what it needs. Background: `.agents/research/public-checkbox-datasets.md`.

| Stage | What is tuned | Train / validation data |
|---|---|---|
| `yolox` | Weights: fine-tune `form-field-v1-nano` from its published `best_ckpt.pth` (20 epochs on tiles cut as the service cuts them; every box is ChoiceButton, so the ONNX output is unchanged), then `YoloxParams` | Synthetic pages + real scans with bbox |
| `cnn` | Weights: fine-tune `form-field-v1-state` from its published `state_cnn.pt` on box crops made by the service's own crop, then `StateCnnParams` | Synthetic marks + real scans with trusted state |
| `opencv`, `fill_ratio` | `OpenCVParams`, `FillParams`, `BinarizeParams` by search (no weights) | Validation split of the same data |
| label | Not tuned (see below); the OCR recognizer is not fine-tuned (no licensed dataset has box + label text) | — |

Sources (classes remapped to `checked` / `unchecked`; licences allow training), as built in `training/data.py`:

- **Synthetic:** the official blank 1004 (Freddie Mac form 70), 1004C (70B) and 1004MC (Fannie Mae, saved by hand: its site answers scripts with a Cloudflare challenge) PDFs. Every checkbox is a stroked vector square, so the bbox is exact. Boxes are marked at random (X, ✓, fill, partial fill, scribbles; empty boxes sometimes crossed by a stray stroke), rendered at 100–300 DPI and degraded like a scan (slight rotation, blur, noise, JPEG, colour watermark).
- **Real scans (CC BY 4.0):** Roboflow `checkboxes-2`, `cogniquest-idc1w/checkboxes-xtv3p` (boxes only: on its crops the `checked` class is mostly empty boxes), `checkbox_v2`, `checkbox-detector` and `government-document-omr-checkbox`, forked so they export at their own size (the published versions are stretched to a square). Web screenshots are dropped by colour. Their boxes are drawn loose (the printed square fills 75–90% of the side), so each is snapped to the printed square inside it.
- **Dropped after looking at their crops:** `tax-form-checkbox-detection` (round radio buttons labelled with their word) and `checkbox-detection-jiid0` (10×10 px crops).
- **Excluded from training:** `annotation-mmvj7/new_merge_project`, whose pages are the external evaluation set; AutoFormBench and CheckboxQA (non-commercial, evaluation only).

Rules:

- Train and validation splits are made **by source form**: the synthetic 1004C is the validation form; real scans have no form id, so pages whose thumbnails look alike are one group and whole groups go to validation. Parameters, thresholds and the training checkpoint are chosen on validation only.
- A field of a `*Params` moves only when it gains at least 0.5 points on validation. Smaller gains are noise of the split: a first search moved `max_aspect` for 0.1–0.4 points and halved OpenCV recall on the external set. The rule was added after seeing that, and applies to every stage.
- `LabelParams` are not tuned. A search against the PDF text layer of synthetic pages chose values that read the samples' labels 7–11 points worse, so its result was not applied.
- A checked box whose mark the clean gray erases (a colour tick) is left out of the `cnn` crops: the model would see an empty box labelled checked, and it learned to call thick-bordered empty boxes checked.
- Evaluation pages (4 samples, external set, user-added pages) are never read by `training/`.
- Training runs locally (PyTorch on Apple MPS). Code lives in `backend/training/`, its own uv project outside `src/`, so the service never imports it and the runtime image never contains PyTorch. Datasets are downloaded by a script, not committed.
- Fine-tuned weights are exported to ONNX with the same input and output as the pretrained model, so `locate/yolox.py` and `classify/state_cnn.py` only change their model file. They are committed to the repo under their own names in `backend/models/` (≈4 MB; only the final version of each stage), while the pretrained weights fetched by `make models` stay as the "before" row of the evaluation tables.
- Tuning still favours detection recall, and resolves an ambiguous state to unchecked.

Results on validation: `yolox` F2 0.779 → 0.942, `cnn` state accuracy 0.955 → 0.982 (pretrained at its best setting), `OpenCVParams` F2 0.748 → 0.810, `FillParams` 0.908 → 0.930. End to end on the external set, `yolox` + `cnn` goes from 57.3 to 71.8 and `yolox` + `fill_ratio` from 45.6 to 70.4; on the samples all combos stay at 96.9–99.0.

## 7. Evaluation

([evaluation ticket](https://github.com/martinnbejarano/checkbox-detection/issues/6))

### Ground truth (the 4 samples)

`backend/tests/data/ground_truth/sample-00N.json` (image paths are relative to the repo root):

```json
{
  "image": "samples/sample-000.png",
  "boxes": [
    { "bbox": [x1, y1, x2, y2], "is_checked": true, "label": "Urban", "tags": ["handwriting"] }
  ]
}
```

Allowed tags: `watermark`, `color_fill`, `handwriting`, `checkmark`, `partial_fill`, `touches_line`.

`label` is the printed text immediately right of the box, as a reader would type it (`null` when there is none). It is used only by the label metric.

**Labeling policy** (applies to the ground truth and to the classifiers' target):

- A box is a printed **square or rectangle**. Round radio buttons are out of scope (listed as a README limitation).
- `is_checked: true` means an **intentional mark mostly inside the box**: an X, a ✓, or a full or partial fill.
- `is_checked: false` covers these cases:
  - a stray stroke that only crosses the box (such as the hand-drawn lines on the cropped sample);
  - watermark ink;
  - colored cell fill;
  - an empty box.

**Labeling procedure.** Claude does the labeling.

1. **Label from the image, not from the pipeline.** Claude labels each sample by looking at zoomed tiles of the page, before running `prelabel.py`. This keeps the ground truth independent of any stage's parameters.
2. **Spot-check (human).** The user reviews `draw_boxes.py` overlays of the ground truth (boxes, state and label text) for all 4 samples and flags errors. The labels are committed only after this review.
3. **Reconcile, later.** Once the pipeline and `prelabel.py` exist, list every disagreement between them and the labels. A label changes only when the image shows it was wrong, never to match the pipeline. Each change is listed in that PR for the user.

Because Claude also builds the pipeline, the main evidence of generalization is the **external set** below. Its labels were made by third parties.

### External evaluation set (never used for tuning or training)

About 15–20 pages from the public-domain Roboflow housing-forms dataset (`annotation-mmvj7/new_merge_project`). They are chosen to differ from the samples in layout and box size, and converted to the same ground-truth JSON with class names remapped to `is_checked` (no `label`). They are stored under `backend/tests/data/external/` with a README giving the source, license and selection rule.

As built: 15 pages, 419 boxes (155 checked), from two form layouts at five image sizes; boxes are 12–31 px. The source annotates each box together with its label text, so each region was shrunk to the printed square at its left end (template match, then every box reviewed on zoomed crops; 49 corrected by hand). Which boxes exist and their state are the third party's. Forms that mark answers by circling a word are excluded (not a printed box).

- Nothing is tuned or trained on the samples or the external set: both are evaluation only (section 6, "Tuning on public data"). Pages the user adds later join them under `backend/tests/data/`.
- The README reports both tables: "samples" and "external", each before and after tuning.
- Choosing the default pipeline is a human checkpoint. The user looks at both tables, broken down by combo, tag and scale. There is no fixed threshold.

### `scripts/evaluate.py`

- Runs every **locator × classifier** combo in `LOCATORS` × `CLASSIFIERS` (or the ones passed with `--locator`/`--classifier`). Combos whose model files are missing are skipped with a note.
- Matching: greedy by IoU, one-to-one, per page.
- Reports, per combo, per image and in total:
  - Detection precision, recall and F1 at IoU 0.5, plus recall at IoU 0.3.
  - State accuracy on matched boxes, with false-checked and false-unchecked counts shown separately.
  - **Headline: end-to-end accuracy**, where a ground-truth box counts only if it was matched at IoU ≥ 0.5 **and** its state is correct.
  - Results broken down by tag.
  - With `--scales`, the same metrics for each sample resized by each factor (scale-robustness table).
  - Latency p50 and p95 per page (in-process; the Docker numbers are measured separately with `curl` timing).
- With `--labels` (samples only, since the external set has no labels): **label accuracy** over matched boxes whose ground truth has a label = share whose OCR text equals the ground truth after case and whitespace normalization, plus OCR latency per page.
- Output: Markdown tables pasted into the README. `make eval` runs every combo with `--scales 0.5,1,1.5 --labels`.

## 8. Tests

([evaluation ticket](https://github.com/martinnbejarano/checkbox-detection/issues/6). The rules for planning and writing tests live in `.claude/skills/testing/`. Every implementation session loads that skill.)

**Principle: "Write tests. Not too many. Mostly integration."** Tests catch regressions that matter; coverage is not a goal. Accuracy is not a unit-test concern: it is measured by `make eval` and locked by `test_golden.py`.

### What gets a test

A test exists only if it protects observable behavior against a realistic regression that no other test catches. Before writing it, the author states **the code edit that would make it fail** ("if the HSV mask in `to_clean_gray` is removed, the red-watermark page returns boxes"). No sentence, no test. Inputs are partitioned into equivalence classes, one test per class.

Not tested: library behavior (Pillow, OpenCV, onnxruntime, RapidOCR, pydantic), `Settings` and `*Params` values, one-liners, and wiring such as `Pipeline.run` calling its stages (every integration test exercises it).

### Levels

| Level | How | What lives here |
|---|---|---|
| **Integration (default)** | `with TestClient(create_app(Settings(locator=..., classifier=...)))`, real pipeline, real model files, synthetic pages drawn with `cv2` or real samples | API contract and `page`; `?labels=true` vs default; every error status as problem+json; box order; multi-page PDFs; golden accuracy per combo |
| **Unit, real dependencies** | One stage called directly with its real model | Cases that would be slow or opaque through HTTP: the scale grid (box side 10–100 px × page width 1000–5000 px), grayscale vs RGB input, boxes touching table lines, tile edges, each mark class for each classifier |
| **Unit, pure** | Plain functions, no dependencies | `iou`, `nms`, `tile_windows`, evaluation matching and label normalization, label-strip geometry |

Fakes: only to force a failure the real code can't produce on demand (the 500 path). No `unittest.mock`, no faked stages to feed data, no assertions on which methods were called.

### Files

| File | Level | What it proves |
|---|---|---|
| `test_api.py` | Integration | 200 shape with `page`; `?labels=true` adds `label` and the default omits it; 413 / 415 / 422 / missing field / 500 as problem+json without stack traces; `/health`; unknown stage name and missing model fail startup; one JSON log line per request with stage timings. |
| `test_pdf.py` | Integration | A 2-page PDF made in the test returns boxes tagged `page` 0 and 1 at 300 DPI; page and pixel caps; encrypted and corrupt PDFs; concurrent PDF requests give consistent results. |
| `test_golden.py` | Integration | Each sample × each (locator, classifier) combo reaches end-to-end accuracy ≥ a floor recorded after tuning, so a regression fails while small tweaks do not. |
| `test_locate.py` | Unit | For each locator: synthetic pages, as grayscale and as RGB (same boxes expected); box sides 10–100 px on pages 1000–5000 px wide (yolox: only the box sizes the model sees, bbox within ±8% of side); bbox within ±3 px or ±5% of side; text-only page → `[]`. OpenCV: box touching a table line, colour overlays. YOLOX: red overlay ignored, box across a tile border found once. YOLOX has no table-line logic of its own, so that case is not tested for it. |
| `test_classify.py` | Unit | For each classifier, one case per class: empty, X, ✓, filled, stray stroke crossing the box (`fill_ratio` only; the CNN calls it checked), near-threshold or ambiguous score → unchecked. |
| `test_label.py` | Unit | `cv2.putText("Urban")` right of a drawn box → "Urban"; nothing to the right → `None`; the strip stops at the next box on the row and at the page edge. |
| `test_geometry.py` | Pure | `iou`, `nms` (nested and cross-tile duplicates), reading order, `tile_windows` covers the page with overlap. |
| `test_imaging.py` | Unit | PNG/JPEG/TIFF/WEBP decode; gray stays 2-D, RGBA on white, 16-bit gray rescaled; EXIF rotation; truncated → `CorruptInput`; non-image → `UnsupportedFormat`; pixel cap before decode. |
| `test_evaluate.py` | Pure | Greedy one-to-one IoU matching on hand-made boxes; label normalization. |

### How a test looks

- Body of 25–30 lines, 45 at most: build this page, call this, assert this outcome.
- DAMP over DRY: helpers hide what is irrelevant (PNG encoding, client setup); box positions, marks and expected states stay visible in the body.
- Assert the whole outcome with hardcoded values (all states, every bbox within tolerance, the full error body). Never compute the expectation with the code under test.
- `pytest.mark.parametrize` only when the body is identical; each case has an `id` naming its class.
- No `skip`/`skipif` and no `if` in a test body: `make test` fetches the model files first, so every test always runs.
- Tests build `Settings(...)` explicitly and never read the environment.
- Golden-standard exemplars to copy are listed in `.claude/skills/testing/references/guidelines.md`.

### When tests run

1. **Before coding a ticket:** the session writes two tables (tests to write, each with its breaking edit; tests to skip, each with its reason). They go in the PR description.
2. **While coding:** focused `pytest -k` runs.
3. **Before opening the PR:** `make check` (lint, typecheck, full suite incl. golden tests). For at least one new test, the author breaks the code on purpose, confirms the test fails, reverts, and says which in the PR.
4. **On every PR:** GitHub Actions runs `make check` (which fetches the pinned models). The PR is merged only when it is green.
5. **Accuracy changes:** `make eval` tables before and after go in the PR whenever a stage or its parameters change.
6. **Bug fixes:** a regression test that fails on `main` and passes with the fix, when it clears the bar above; otherwise the PR says what evidence was used instead.

## 9. Runtime and performance

- Endpoint declared with plain `def`. FastAPI runs it in its threadpool. OpenCV and onnxruntime release the GIL during heavy calls. pdfium calls are serialized by the lock.
- The pipeline is built once in `lifespan` (`build_pipeline(...)`), stored on `app.state`, and injected through a `get_pipeline` dependency. Only the models the config selects are loaded. A missing model file stops startup before `/health` turns green.
- ONNX sessions are created only by `detection/onnx.load_session`: `intra_op_num_threads = onnx_threads` (default 1), `inter_op_num_threads = 1`, CPU provider. One session per model is shared by all requests (onnxruntime allows concurrent `run` calls on one session).
- `cv2.setNumThreads(settings.cv_threads)` is set at startup (default 1 per worker), to avoid CPU oversubscription when there are several workers. Throughput is scaled with uvicorn workers or replicas, never with a process pool inside the app.
- Logged per request (JSON lines): request id, status, total ms, and also `pages`, `decode_ms`, `locate_ms`, `classify_ms`, `label_ms` and `n_boxes`. The response carries an `X-Request-ID` header.
- **No fixed latency target.** `evaluate.py` measures p50/p95 per combo and the README reports them. Probe numbers on one M4 thread (a Railway vCPU is about 2–3× slower, est.): YOLOX ≈ 250 ms/page, state CNN ≈ 0.2 ms/box, label OCR ≈ 9 ms/box.
- Memory: all selected models together stay near 0.5 GB peak (probe/est.), so 2 GB is enough. No PyTorch at runtime.

## 10. Configuration (`config.py`)

`Settings(BaseSettings)`, env prefix `CHECKBOX_`. The defaults run with no env vars; the fine-tuned model files they load are committed in `backend/models/`.

| Setting | Default |
|---|---|
| `locator` | `"yolox"` (`"opencv"` needs no model files) |
| `classifier` | `"cnn"` (`"fill_ratio"` needs no model files) |
| `models_dir` | `models` (relative to `backend/`); `/app/models` in the Docker image |
| `onnx_threads` | 1 |
| `max_upload_bytes` | 20 MiB |
| `max_image_pixels` | 50_000_000 |
| `max_pdf_pages` | 20 |
| `pdf_dpi` | 300 |
| `cv_threads` | 1 |
| `log_level` | `"INFO"` |
| `static_dir` | unset (no frontend mount); `/app/static` in the Docker image |

Stage tuning parameters are **not** settings. They live in frozen dataclasses next to their code (`OpenCVParams`, `YoloxParams`, `FillParams`, `StateCnnParams`, `LabelParams`).

## 11. Tooling and code conventions

- **Python 3.13 and uv** (`uv.lock` committed). Runtime dependencies: `fastapi`, `uvicorn`, `pydantic-settings`, `python-multipart`, `opencv-python-headless`, `numpy`, `pillow`, `pypdfium2`, `onnxruntime`, `rapidocr`. Dev dependencies: `pytest`, `httpx2` (Starlette's TestClient deprecates `httpx`), `ruff`, `mypy`. `huggingface_hub` is never a dependency: `make models` fetches each pinned file with `curl` from `huggingface.co/<repo>/resolve/<sha>/<file>`. PyTorch is never a dependency of the service.
- **ruff** for linting (rule sets E, F, I, B, UP, SIM, RUF) and formatting. **mypy --strict** on `backend/src/`.
- **Make targets:** one root `Makefile`, which is the only entry point. Backend targets run `uv --directory backend …`, and frontend targets run `npm --prefix frontend …`. The targets are `install`, `models`, `lint`, `typecheck`, `test`, `eval`, `run`, `docker-build`, `docker-run`, `frontend-dev` and `frontend-build`. `make models` downloads the files pinned in `backend/models.txt` into `backend/models/<repo name>.onnx` (skips files already present); `make test` depends on it, so tests never skip for missing models. `make check` runs lint, typecheck and test for the backend, plus the frontend build (`tsc` and oxlint) once `frontend/` exists.
- **Conventions:**
  - Functions are small and named for what they return (`find_candidates`, `fill_ratio`).
  - Full type hints everywhere.
  - No magic numbers outside the `*Params` dataclasses and `Settings`.
  - Each module docstring states its responsibility.
  - Comments explain *why*, not *what*.
  - `TODO(prod):` marks every known production gap, so `grep TODO(prod)` gives the list in the writeup.
- **Do not build:** job queues, a frontend beyond the demo page in section 13, process pools, a database or cache, API versioning, auth, a metrics stack, k8s manifests, image fetch by URL, Protocols beyond `Locator` and `StateClassifier`, ABCs, factories or plugin registries, runtime model downloads, full-page OCR, a VLM stage, deskew (add only if evaluation shows rotated pages failing), or a config file format. Each of these appears as a `TODO(prod)` only where it would really go.

## 12. Packaging and deployment

([deployment ticket](https://github.com/martinnbejarano/checkbox-detection/issues/9))

- **Dockerfile:** `backend/Dockerfile`, multi-stage, following uv's official recipe. It is built from the repo root, `docker build -f backend/Dockerfile .`, because the image also contains the built frontend.
  - Frontend stage: `node:22-slim`, which runs `npm ci && npm run build` in `frontend/` and produces `frontend/dist`, which is copied to `/app/static`.
  - Models: the default pipeline runs only the fine-tuned `*-tuned.onnx` files, which are committed (§6), so they are copied from `backend/models/` into `/app/models`. The pretrained files in `backend/models.txt` are only for training and evaluation and stay out of the image. RapidOCR's recognizer ships inside its wheel.
  - Builder: `python:3.13-slim` plus a pinned `ghcr.io/astral-sh/uv`, running `uv sync --locked --no-dev --no-editable`, dependencies first and then the project, so the package is installed into the venv.
  - Runtime: `python:3.13-slim` with only the venv, `/app/static` and `/app/models`, plus `CHECKBOX_STATIC_DIR=/app/static` and `CHECKBOX_MODELS_DIR=/app/models`. It runs as a non-root user with an exec-form `CMD ["uvicorn", "checkbox_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
  - `HEALTHCHECK` via python/urllib on `/health`.
  - `backend/Dockerfile.dockerignore` excludes `.agents/`, `samples/`, `backend/tests/`, `backend/training/`, `backend/scripts/`, `backend/models/` except `*-tuned.onnx`, `.git`, `backend/.venv`, `frontend/node_modules` and `frontend/dist`.
  - No apt packages are needed: OpenCV is the headless wheel, pypdfium2 ships its own binary, and onnxruntime and RapidOCR are pip wheels.
- **Local (primary path):** `make docker-build && make docker-run`, or `make install && make models && make run` without Docker.
- **Railway (demo URL):** the same image on CPU (1 vCPU / 2 GB), built using the root `railway.toml` (`dockerfilePath = "backend/Dockerfile"`), with the service variable `PORT=8000` so Railway routes to the port the exec-form `CMD` listens on. If the measured latency is too slow, the next step is more vCPUs; moving models to a hosted GPU (e.g. Hugging Face) is a separate decision with its own ticket. The README states the URL, the limits, and that it is a demo instance. `TODO(prod)`: auth and rate limiting in front of the public URL.

## 13. Demo frontend (`frontend/`)

([frontend ticket](https://github.com/martinnbejarano/checkbox-detection/issues/23))

One page to try the API by hand. It is the public face of the Railway demo and a tool for checking the pipeline's output, so it gets a real design (empty, loading and error states included). All UI copy is in English.

- **Stack:** React, Vite and TypeScript, made from the official `react-ts` template. No UI library, no router and no state library. The only extra dependencies are `pdfjs-dist`, to render PDF pages, and `@fontsource-variable/inter`, the self-hosted UI face. The browser's own `fetch` sends the request.
- **Behavior (one screen):**
  1. Drag and drop, or click to pick, an image or a PDF. It is sent at once to `POST /detect?labels=true` as `FormData`.
  2. Two columns:
     - **Document** (left): the image, or the current PDF page rendered with pdf.js, with a page selector for multi-page PDFs. Each box is drawn on top: green when checked, red when unchecked, orange (dashed) when `score` is below `LOW_CONFIDENCE` (0.3, where YOLOX's real boxes start to thin out). Coordinates are scaled from the image's natural size (for PDFs, the page at 300 DPI) to its displayed size.
     - **Detections** (right): one row per box with its label and state. A box without a label still gets a row that says the label was not detected.
  3. Linking the two: hovering a row draws a connector line from the row to its box and highlights the box; hovering a box highlights its row. Clicking a row scrolls the document to the box. If the box is out of view the line points to the edge of the document pane. No zoom or pan.
  4. Errors (4xx/5xx) show the problem+json `title` and `detail`.
  5. Desktop first. On narrow screens the columns stack and the connector line is not drawn.
- **Out of scope:** the raw JSON body, count summaries, filters and built-in sample pickers.
- **API types:** `frontend/src/api.ts` has `detect(file): Promise<DetectResponse>` plus the response types, written by hand to match the backend's `api/schemas.py`.
- **How it reaches the API:**
  - In development, `npm run dev` runs on Vite's port, and `vite.config.ts` proxies `/detect` to `http://localhost:8000`. No CORS setup is needed.
  - In Docker and on Railway, FastAPI serves the built frontend with `StaticFiles` at `/`, from the directory in the `static_dir` setting. The mount is added **after** the API routes, so `/detect` and `/health` take priority. It is one container and one origin, so again no CORS. When `static_dir` is unset (local backend-only runs), there is no mount. The backend never refers to `frontend/` by path.
- **Quality bar:** the `react-ts` template's linter (oxlint) and `tsc` pass; `make frontend-build` runs `npm run lint` and `npm run build`. No frontend unit tests. One manual check is listed in the README.
- **Make targets:** `make frontend-dev` and `make frontend-build`.

## 14. README (the writeup)

Sections:

1. Quick start (Docker, local incl. `make models`, Railway URL, the demo page at `/`, and `curl` examples with and without `?labels=true`).
2. API contract (including the `page` and `label` extensions and the error table).
3. Approach and why: a 3-stage pipeline (locate → classify → label), classical baseline next to pretrained ONNX models, measured, and swappable per stage.
4. Results tables from `evaluate.py`: per combo, samples vs external, by tag, by scale, latency, and label accuracy.
5. Alternatives considered and why they were not chosen: managed services (privacy, credentials, cost); VLMs (coarse coordinates on small boxes, minutes per page on CPU); Docling layout, FFDNet and `wendys-llc/checkbox-detector` (probe results); full-page OCR (time and memory); PaddleOCR / docTR / Tesseract; training from scratch.
6. Tradeoffs and known limitations. These include: tuning data that are not appraisal scans (synthetic 1004 pages and other forms); box styles never seen in the samples (rounded corners, circles, dashed borders); labels read only to the right of the box; the new URAR form replacing the 1004 family from 2026-11-02; first TIFF frame only; and the cap at 20 PDF pages.
7. Next steps: more real appraisal scans for training, labels to the left and row questions, reading AcroForm values from born-digital PDFs.
8. `TODO(prod)` list.

## 15. Build order

**Workflow:** one GitHub ticket per vertical slice. Each ticket is built in a fresh `/implement` session on its own branch, opens a PR against `main` that links the ticket, and is merged by the user. There are three human checkpoints:

1. ground-truth spot-check;
2. default-pipeline decision;
3. approval before the Railway deploy (the user has an account; the deploy uses the Railway CLI).

Order (done: scaffold, walking skeleton, upload validation):

0. CI: GitHub Actions running `make check` on every PR.

1. Refactor `detection/` into the 3-stage layout with OpenCV behind it (no behavior change).
2. In parallel: multi-page PDF; OpenCV locator + `fill_ratio` on real forms; YOLOX locator + CNN classifier; label stage; ground truth; demo frontend.
3. `evaluate.py` over all combos, then tune every stage on public data (fine-tune `yolox` and `cnn`) and lock with golden tests.
4. Choose the default pipeline from the numbers.
5. Dockerfile (frontend and models stages), then the Railway deploy.
6. README with the measured results.
