# checkbox-detection

An HTTP service that finds the checkboxes on a document page, says whether each one is checked and, on request, reads the label printed next to it. 

- Demo page: [https://homevision.martinbejarano.com](https://homevision.martinbejarano.com)
- API: [https://homevision.martinbejarano.com/detect](https://homevision.martinbejarano.com/detect) (interactive docs at [`/docs`](https://homevision.martinbejarano.com/docs))

## Quick start

### Docker

```sh
make docker-build   
make docker-run      
```

Open [http://localhost:8000](http://localhost:8000) for the demo page. The image holds the API, the built frontend and the two fine-tuned models, so it needs nothing else.

### curl

```sh
curl -F file=@samples/sample-003.png http://localhost:8000/detect
```

```json
{"boxes": [
  {"bbox": [455, 397, 489, 430], "is_checked": true,  "page": 0, "score": 0.869},
  {"bbox": [590, 398, 621, 431], "is_checked": false, "page": 0, "score": 0.905}
]}
```

With labels:

```sh
curl -F file=@samples/sample-003.png "http://localhost:8000/detect?labels=true"
```

```json
{"boxes": [
  {"bbox": [455, 397, 489, 430], "is_checked": true,  "page": 0, "score": 0.869, "label": "Owner"},
  {"bbox": [590, 398, 621, 431], "is_checked": false, "page": 0, "score": 0.905, "label": "Tenant"}
]}
```

Swap `http://localhost:8000` for the demo URL to try the deployed instance.

### Demo page

Open the demo page ([http://localhost:8000](http://localhost:8000) with Docker, or the URL Vite prints) and drop a form on it, or pick one of the four sample pages under the drop target.

1. The page appears with a mark on every box: hatched green for checked, open red for unchecked, dashed orange for low confidence (`score` &lt; 0.3). The Detections list fills on the right.
2. Hover a row: a leader line joins it to its box, the box is circled, and its extent shows on the rulers. Hover a box: its row highlights. Click a row: the page scrolls to the box.
3. Drop a multi-page PDF: each page's detections appear as soon as that page is read, while the rest are still processing. The pager switches pages and the list follows.

## How it works

Each page goes through four stages, in this order:

1. Straighten: turn the page upright and level (`detection/orient.py`, `detection/deskew.py`). The boxes are mapped back to the original image's pixels at the end.
2. Locate: where are the boxes (`detection/locate/`).
3. Classify: is each box checked (`detection/classify/`).
4. Label: what text sits to its right (`detection/label.py`), only when `?labels=true`.

`backend/src/checkbox_detection/detection/__init__.py` is the file to read first. `Pipeline.detect` calls the stages by name, so the order is visible in one place.

Locate and classify have two implementations each, and `CHECKBOX_LOCATOR` / `CHECKBOX_CLASSIFIER` pick one. Each is one module plus one entry in a dict, and `evaluate.py` loops over the same dicts to compare every combination.


| Stage      | Implementation                | What it does                                                                                                                                                                                                                                                                                                                |
| ---------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Straighten | RapidOCR + projection profile | RapidOCR finds the page's text lines. When most are taller than wide the page is on its side and gets a quarter turn; a tilt up to ±10° is then levelled by projection profile; and when the widest lines read better turned 180° the page is upside down.                                                                  |
| Locate     | `opencv`                      | Classical baseline. Removes saturated colour (the red watermark, blue cell fill), binarizes, erases long table rules, then keeps closed, hollow, square-ish contours. The page sets its own box size: weak candidates are kept only near the median side of the confident ones, which drops square glyphs like "O" and "D". |
| Locate     | `yolox` (default)             | `nutrientdocs/form-field-v1-nano`, a 0.9M-parameter YOLOX-Nano (Apache-2.0), fine-tuned. The page is cut into overlapping tiles whose size is a fraction of the page width, each tile runs through the model, and NMS merges the results.                                                                                   |
| Classify   | `fill_ratio`                  | Ink share inside the box, after shrinking past the border. Strokes that sit mostly outside the box do not count, so a line crossing an empty box stays unchecked.                                                                                                                                                           |
| Classify   | `cnn` (default)               | `nutrientdocs/form-field-v1-state`, a 96k-parameter CNN on 40×160 grey crops (Apache-2.0), fine-tuned. All crops of a page go through one batched run.                                                                                                                                                                      |
| Label      | RapidOCR                      | Recognition only (no text detection) on a strip cut right of each box. The strip ends at the next box on the row, a vertical rule, or a blank gap after the text.                                                                                                                                                           |


Why it is built this way:

- Splitting locate from classify lets each part be measured and swapped alone. A detector that also predicts the state has to be right about both at once, and you can't tell which half failed.
- The classical pipeline came first because it needs no data or weights and every threshold can be read. It stays as the baseline and as the fallback that runs without model files.
- The learned models are tiny pretrained ONNX files that run on onnxruntime, so the service needs no PyTorch at runtime.

### Tuning

The 4 original sample pages are too few to tune on, so every stage was tuned on public data and the samples are only used to evaluate.

- Training data: synthetic pages made from the official blank 1004, 1004C and 1004MC PDFs (boxes marked at random, rendered at 100 to 300 DPI and degraded like a scan) plus five CC BY 4.0 Roboflow checkbox datasets. 3,205 pages and 48,414 boxes for training, 610 pages and 10,426 boxes for validation. The split is by source form: the synthetic 1004C is the validation form.
- `yolox` and `cnn` were fine-tuned from their published checkpoints and exported back to ONNX with the same inputs and outputs, so only the model file changed.
- The `*Params` of every stage were chosen by search on validation. A value only moved when it gained at least half a point. 
- `training/` is its own uv project with PyTorch. The service never imports it, and it never reads the samples or the external set.

## API

### `POST /detect`

The request is `multipart/form-data` with one field, `file`. It accepts PNG, JPEG, TIFF (first frame only), WEBP and PDF. The format is read from the bytes; the `Content-Type` header and the filename are ignored.

The query parameter `labels` (bool, default `false`) adds a `label` to each box.

The response is `200` with `{"boxes": [...]}`, and an empty list when nothing is found. Each box has:


| Field        | Meaning                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bbox`       | `[x1, y1, x2, y2]`, integer pixels of the top-left and bottom-right corners, in the upright image (after EXIF rotation). For PDFs, the page rendered at 300 DPI.    |
| `is_checked` | `true` for an intentional mark mostly inside the box (X, ✓, full or partial fill). A stray stroke crossing the box, a watermark or a coloured cell fill is `false`. |
| `page`       | 0-based page index; always `0` for images.                                                                                                                          |
| `score`      | The locator's confidence that this is a checkbox, in \[0, 1\]. The OpenCV locator has none and always returns `1.0`.                                                |
| `label`      | Only with `?labels=true`. The text read right of the box, or `null` when nothing was read.                                                                          |


Boxes come sorted by page, then top to bottom, then left to right.

`bbox` and `is_checked` are the original contract. `page`, `score` and `label` are extra fields, so a client that ignores them still gets that shape.

### `POST /detect/stream`

The same request and the same boxes as `/detect`, sent page by page so a client can show the first pages while the rest are still being read. The response is `application/x-ndjson`: one line per page, in page order, each sent as soon as that page is done.

```json
{"page": 0, "boxes": [{"bbox": [455, 397, 489, 430], "is_checked": true, "page": 0, "score": 0.869}]}
{"page": 1, "boxes": []}
```

Every upload check runs before the first line, so a rejected upload gets the same problem+json error as `/detect`. A failure after the first line ends the stream with one `{"error": "..."}` line. The demo page uses this endpoint.

### Settings

Environment variables with the prefix `CHECKBOX_`. The defaults need none.


| Variable                                       | Default                                                       |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `CHECKBOX_LOCATOR`                             | `yolox` (or `opencv`, which needs no model file)              |
| `CHECKBOX_CLASSIFIER`                          | `cnn` (or `fill_ratio`, which needs no model file)            |
| `CHECKBOX_MODELS_DIR`                          | `models`, relative to `backend/`                              |
| `CHECKBOX_ONNX_THREADS`, `CHECKBOX_CV_THREADS` | 1 each; scale with uvicorn workers instead                    |
| `CHECKBOX_MAX_UPLOAD_BYTES`                    | 50 MiB                                                        |
| `CHECKBOX_MAX_PDF_PAGES`                       | 100                                                           |
| `CHECKBOX_MAX_IMAGE_PIXELS`                    | 50,000,000                                                    |
| `CHECKBOX_PDF_DPI`                             | 300                                                           |
| `CHECKBOX_LOG_LEVEL`                           | `INFO`                                                        |
| `CHECKBOX_STATIC_DIR`                          | unset; the Docker image sets it to serve the demo page at `/` |


## Repository

- `backend/`: the service (`src/`), offline tuning (`training/`), `scripts/evaluate.py`, tests and the Dockerfile.
- `frontend/`: the demo page (React, Vite, TypeScript).
- `samples/`: the pages used for evaluation, with ground truth in `backend/tests/data/ground_truth/`.