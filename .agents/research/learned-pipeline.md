# Learned 3-stage pipeline: locate → classify state → OCR label

Researched 2026-10-09/10. Scope: redesign `POST /detect` as (1) locate checkboxes, (2) classify checked/unchecked on
crops, (3) OCR the label text next to each box. Constraints: CPU only, Docker on Railway (1 vCPU, 1-2 GB RAM),
Python 3.13, uv, no apt packages, permissive licenses (public API). This note fills gaps left by
`deep-learning-and-data.md` (YOLO26/RF-DETR/D-FINE/FFDNet basics and datasets are covered there, not repeated here).

Bracketed numbers like [3] point to the Sources section. Claims marked **(est.)** are my own estimates. Rows marked
**(probe)** come from a zero-shot run I did on the 4 `samples/` with onnxruntime 1.31.0, 1 intra-op thread, Apple M4
(scratch venv, not committed). There is no ground truth yet, so probe results are counts plus visual inspection of
the top half of each page, not P/R. Expect a Railway vCPU to be about 2-3x slower than one M4 core **(est.)**.

---

## TL;DR

- **Best new find for stage 1+2: `nutrientdocs/form-field-v1-nano` + `form-field-v1-state`** [20][21]. Both are
  Apache-2.0, ONNX, and run with onnxruntime alone. The detector is YOLOX-Nano (Apache-2.0 base, so there is no
  Ultralytics/AGPL taint): 0.9M params, 3.7 MB, 640 px input. The state model is a 96k-param CNN on 40×160 gray
  crops that outputs empty or filled. **(probe)** Tiled at 1280-px windows, nano found 73 choice boxes on
  sample-003 in 222 ms. The state model labeled them in 17 ms, and every box I checked by eye in the top half was
  right. On sample-000, where boxes are bigger (~45 px), recall was clearly lower, so tile scale needs tuning. Use
  it as a strong baseline and teacher, not a finished answer.
- **Docling layout models (heron/egret) do have `checkbox_selected` / `checkbox_unselected`,** under Apache-2.0
  weights and MIT code [1][2][3]. They are a poor fit for us, though. Input is a fixed 640×640 with an aspect
  squash. Only 3,977 selected and 57,348 unselected training instances exist (about 2.6% of 2.3M), and no
  per-class AP is reported [4]. **(probe)** On our pages heron's boxes cover the checkbox *plus its label word*,
  not the square. It missed many empty boxes, and its state was often wrong (for example, "☒ Fee Simple" came out
  unselected). Latency was 360 ms/page at 1 thread. The ONNX export weighs 171 MB, plus 400-600 MB RSS.
- **FFDNet ONNX exists** (`jbarrow/FFDNet-{S,L}-cpu`, fixed 1216×1216) and runs with plain onnxruntime [6][7]. Its
  **ONNX metadata says `license: AGPL-3.0` (Ultralytics YOLO11)**, and the model cards and the GitHub repo still state
  no license [5][8]. Treat it as AGPL. **(probe)** FFDNet-S took about 195 ms/page and found mostly empty boxes,
  missing many ☒ (it was trained on empty widgets). FFDNet-L ONNX returned **0 boxes on 3 of 4 samples** (max score
  0.004 on sample-003).
- **`wendys-llc/checkbox-detector`** outputs checked/unchecked directly [22]. Its card says Apache-2.0, but the ONNX
  metadata says **AGPL-3.0, Ultralytics YOLO12n**. Under Ultralytics' terms that is AGPL. **(probe)** Without its
  natural-pdf tiling and magnify logic, recall on ☒ was low.
- **Stage 2 recommendation:** a tiny custom CNN (~50-100k params), or start from `form-field-v1-state`. Train
  on CPU in minutes, export with `torch.onnx.export`, and run in under 0.3 ms/crop. MobileNetV3/ResNet18 are
  overkill. Use 3 classes (`unchecked` / `checked` / `ambiguous`), or 2 classes plus a confidence band.
- **Stage 3: RapidOCR (Apache-2.0, ONNX, pip-only, PP-OCRv6 small models bundled) in recognition-only mode on
  label crops.** **(probe)** That took about 9 ms/crop and 258 MB peak. Full-page OCR took **~10 s and 2.5 GB peak
  RSS** at 1 thread, which does not fit Railway. Avoid PaddleOCR 3.x (pins `numpy<2.4`, which conflicts with our
  2.5.3, and adds a 195 MB paddle wheel), docTR (torch), and Tesseract (needs tessdata; apt is not allowed).
- **Born-digital PDFs:** pypdfium2 (already a dependency) gives text with coordinates and AcroForm checkbox state
  (`FPDFAnnot_IsChecked`). I verified this on a generated fillable PDF. PyMuPDF is AGPL-3.0 or commercial [16],
  so do not use it.
- **Runtime budget:** onnxruntime is a 24 MB wheel. torch CPU is 190 MB+ and installs to roughly 0.7-1 GB (est.).
  A process with nano + state + RapidOCR-rec stays near 0.5 GB peak **(probe/est.)**. Heron, FFDNet-L or
  full-page OCR each push toward or past 1 GB.

---

## 1. Summary comparison

### Stage 1 (locate) candidates

| Model | Classes | Weights / code license | Input | Size | CPU latency/page (probe, 1 thr M4) | Peak RSS (probe) | Notes |
|---|---|---|---|---|---|---|---|
| **form-field-v1-nano** (YOLOX-Nano) [20] | Text, ChoiceButton, Signature | **Apache-2.0** / YOLOX Apache-2.0 | 640² letterbox, decode baked in | 3.7 MB (1.9 MB fp16) | 30-40 ms full page; 220-290 ms tiled at 1280 | ~0.7 GB for nano+state+wendys together with 4200-px pages loaded | Full page at 640 found 0 boxes on sample-003. Tiling is required. Card mAP50-95 on filled forms is 0.550 vs FFDNet-L 0.285 [20]. |
| **Docling heron** (RT-DETRv2-R50) [1][2] | 17 layout classes incl. checkbox_selected/unselected | **Apache-2.0** weights; docling MIT; docling-ibm-models MIT | 640×640, no pad (aspect squashed) [1] | 171 MB (ONNX) | 360 ms full page; 3.3-4.4 s tiled | 411 MB loaded, ~600 MB after run | Box = checkbox + label. Unreliable state. See §2. |
| Docling egret-m/l/x (D-FINE) [3][4] | same 17 | Apache-2.0 | 640×640 | 19.5-62.7M params | Paper: 0.33 s (m) to 0.81 s (x) on EPYC, 4 threads [4] | n/a | No ONNX on HF. transformers `DFineForObjectDetection`. |
| FFDNet-S / -L ONNX [6][7] | textbox, choice_button, signature | **AGPL-3.0 (in ONNX metadata)**; card/repo: none | fixed 1216² | 38 / 102 MB | S ~195 ms; L ~600 ms | S ~0.7 GB; L ~1.1 GB | L returned 0 boxes on 3/4 samples. S misses many ☒. |
| FFDetr (RF-DETR-M) [9] | same 3 | **Apache-2.0** (card) | RF-DETR Medium (576) | 135 MB `.pth` | not measured (needs torch + `rfdetr`) | n/a | Card baselines: Empty 0.461 / Filled 0.170 mAP50-95 [21]. |
| wendys-llc/checkbox-detector (YOLO12n) [22] | checkbox_checked, checkbox_unchecked | Card Apache-2.0; **ONNX metadata AGPL-3.0 (Ultralytics)** | 1024², trained on 72-DPI renders + SAHI tiles | 10.3 MB | ~130 ms (single 1024 pass) | — | Card: P 0.962 / R 0.948, checked recall 0.878 on 102 gold pages [22]. Low ☒ recall in probe without its tiling logic. |
| form-field-v1-small/medium [21] | 3 | **Commercial (Nutrient)**, weights not downloadable | — | 36 MB+ | card: ~11 pages/s CPU | — | Unusable. |

### Stage 3 (OCR) candidates

| Engine | License | Install (pip, Linux x86_64 cp313) | Torch/Paddle? | CPU latency (probe unless cited) | Notes |
|---|---|---|---|---|---|
| **RapidOCR 3.10** [12][13] | Apache-2.0 code; models PaddleOCR (Baidu copyright, Apache-2.0 repo) | `rapidocr` 27 MB wheel (bundles PP-OCRv6 det/rec small ONNX, 30 MB) + `onnxruntime` 24 MB; pulls `opencv-python-headless` (shared with us), shapely, pyclipper, omegaconf | No | Rec-only: ~9 ms per label crop (100 crops 0.88 s, 258 MB peak). Full 2550×3301 page: ~10 s, 224 lines, **2.5 GB peak**. Det-only page: 0.74 s, 1.5 GB peak. | Good English on printed forms in the probe ("Other (describe)", "Census Tract"). Some errors on watermark/shading ("Teunant", "A#0e"). |
| PaddleOCR 3.7 [14] | Apache-2.0 | `paddlepaddle` 195 MB wheel + `paddlex[ocr-core]` (pandas, modelscope, ...) | Paddle | not measured | **`paddlex` pins `numpy<2.4`, which conflicts with our numpy 2.5.3.** PP-OCRv6 released 2026-06 [15]. Same models are reachable via RapidOCR. |
| docTR 1.1 [17] | Apache-2.0 | requires `torch`, `torchvision`, `opencv-python`, h5py ... | Torch | ~1.29 s/page FUNSD (i7-14700K) [18] | Too heavy. |
| OnnxTR 0.9 [18] | Apache-2.0 | onnxruntime + scipy, pyclipper, rapidfuzz...; Python ≥3.11 | No | 0.57 s/page FUNSD (fp32), 0.17 s with mobilenet det+rec (i7-14700K) [18] | Viable alternative to RapidOCR. Models are docTR's (Apache-2.0). |
| Tesseract (tesserocr 2.11) [19] | Apache-2.0 engine; MIT wrapper | tesserocr manylinux wheel 5.5 MB; **tessdata `.traineddata` must be supplied separately** [19] | No | 0.50 s/page FUNSD via pytesseract [18] | `pytesseract` needs the `tesseract` binary (apt). Bundling tessdata yourself is possible but awkward. Weakest on small, noisy text (est.). |

**Pick:** form-field-v1-nano (tiled) → our own tiny state CNN (seeded from `form-field-v1-state`) → RapidOCR
recognition-only on a label strip. All of it runs on onnxruntime. Keep the OpenCV detector as fallback and baseline.

---

## 2. Docling layout model, in detail

- **Current models** (HF org `docling-project`, formerly `ds4sd`): `docling-layout-heron` (default since Docling
  2.50, RT-DETRv2-R50, 42.9M params), `heron-101` (R101, 76.7M), `egret-medium/large/xlarge` (D-FINE + HGNetV2,
  19.5/31.2/62.7M), `docling-layout-old` (RT-DETRv1) [1][3][4]. The only official ONNX is
  `docling-layout-heron-onnx` (171 MB, Apache-2.0) [2].
- **Classes (verified in `config.json`):** caption, footnote, formula, list_item, page_footer, page_header, picture,
  section_header, table, text, title, document_index, code, **checkbox_selected (13), checkbox_unselected (14)**,
  form, key_value_region [1].
- **License.** Weights are Apache-2.0 (all cards) [1][2][3]. `docling` is MIT. `docling-ibm-models` 4.0.3 is MIT
  [10]. The paper says all checkpoints and code are permissive [4].
- **Input resolution.** `preprocessor_config.json`: `size 640×640`, `do_pad: false`. A 2550×3300 page is squashed
  to 640×640 (x scale 0.25, y scale 0.19), so a 20-55 px box becomes about 4-14 px. Training used 640×640 resizes of
  150-DPI pages [1][4].
- **Small objects.** The paper reports only aggregate mAP and AP by size, with no per-class checkbox AP [4]. Training
  checkbox counts: Selected 3,977 (3,071 train), Unselected 57,348 (45,260 train). The original DocLayNet has no
  checkbox classes [4].
  **(probe)** At full page it found 44-92 "checkboxes" per sample, sized 24-99 px. The boxes enclose checkbox plus
  label word ("☐ Vacant"), and the state was often wrong. Tiling at 1280 roughly doubled detections but took
  3-4 s/page.
- **CPU latency.** Paper: heron 0.643 s/image mean on an AMD EPYC 7763 with 4 threads and batch 32. egret-m is
  0.334 s [4]. **(probe)** ONNX at 1 thread on M4: about 360 ms/page.
- **Without the `docling` package.** Yes. The model card runs it with plain `transformers`
  (`RTDetrV2ForObjectDetection` + `RTDetrImageProcessor`) [1], or with onnxruntime only via the heron ONNX, whose
  inputs are `images uint8 [B,3,640,640]` and `orig_target_sizes int64 [B,2]`, and whose outputs are
  `labels/boxes/scores [B,300]` (postprocessing is in the graph) **(probe)**. `docling-ibm-models` depends on
  torch, torchvision and transformers, and **pins `opencv-python-headless<5`, which conflicts with our opencv
  5.0** [10].
- **Install size.** The transformers path needs torch (CPU wheel ~190 MB download, est. 0.7-1 GB installed). The
  ONNX path needs onnxruntime (24 MB) plus the 171 MB model.
- **Verdict.** Not suitable as the stage 1 locator. It may be useful for `form` / `key_value_region` context, which
  we do not need.

## 3. FFDNet / CommonForms updates

- **ONNX:** `jbarrow/FFDNet-S-cpu` (38 MB) and `jbarrow/FFDNet-L-cpu` (102 MB) exist, with a fixed input of
  `1×3×1216×1216` float and raw YOLO output `[1,7,30324]` (cx, cy, w, h + 3 classes, no NMS) [6][7]. The commonforms
  code uses them only in `--fast` mode, "since that's all ONNX supports" (1216) [8].
- **onnxruntime only:** yes. Letterbox to 1216, divide by 255, decode and NMS yourself (about 15 lines).
- **License:** still none on the model cards. The GitHub repo has no LICENSE file (`/LICENSE` returns 404, and the API
  reports `license: null`). The PyPI package has no license field [5][8]. **The ONNX `custom_metadata_map` contains
  `license: AGPL-3.0 License (https://ultralytics.com/license)`, `description: Ultralytics YOLO11{s,l} model`**
  **(probe)**. Conclusion: AGPL. `commonforms` 0.2.1 depends on `ultralytics`, `rfdetr` and `transformers` [8].
- **Apache alternative from the same author:** `jbarrow/FFDetr` (RF-DETR Medium, Apache-2.0 card), `.pth` only, so
  it needs torch and `rfdetr` [9].
- **CPU latency (probe, 1 thread):** FFDNet-S at 1216 took about 190-205 ms. FFDNet-L at 1216 took about 585-610 ms.
  The paper/card default of 1600 px with TTA (`.pt`) would cost roughly 2-4x more **(est.)**.
- **Behavior on our scans (probe):** S gives tight boxes on empty squares but misses many ☒. L gives 0 detections on
  3 of 4 pages. Third-party benchmark: FFDNet-L mAP50-95 is 0.373 on empty, 0.285 on filled and 0.285 on
  handwritten forms [20][21]. That confirms the filled-form weakness.

## 4. Other ready-to-use checkbox/form-field detectors

| Repo | What | License (verified where) | Usable in a public API? |
|---|---|---|---|
| `nutrientdocs/form-field-v1-nano` [20] | YOLOX-Nano detector, 3 classes, ONNX + `.pth` + `exp.py` | Apache-2.0 card. YOLOX base Apache-2.0. Trained on synthetic renders, no PII | **Yes** |
| `nutrientdocs/form-field-v1-state` [21b] | 96k-param depthwise-separable CNN, 40×160 gray letterbox, empty/filled; card acc 0.981 vs ink heuristic 0.915 (synthetic split) | Apache-2.0 | **Yes** |
| `nutrientdocs/form-field-v1-benchmark` [21c] | Eval set (empty/filled/handwritten) | see dataset card | Eval only |
| `wendys-llc/checkbox-detector` [22] | YOLO12n, checked/unchecked, 1024 ONNX, ~5.1k pages (DocumentCloud + CommonForms) | Card Apache-2.0, **ONNX metadata AGPL-3.0** | Risky (Ultralytics says trained models are AGPL) |
| `jbarrow/FFDetr` [9] | RF-DETR-M, 3 classes | Apache-2.0 | Yes, but torch + rfdetr |
| `takumi123xxx/pdfme-form-field-detector` [23] | YOLO11 `best.pt` | Card Apache-2.0; Ultralytics-trained, so AGPL in practice | Risky |
| `yettycz/checkboxes-detector-model`, `wendys-llc/checkbox-detection` | Keras / unnamed blobs, empty cards | MIT tag, no docs | No (no provenance) |
| `hermitkk/omr-checkbox-classifier` | classifier | AGPL-3.0 | No |

## 5. Stage 2: small state classifier

- **Input.** Crop the detected box with 10-25% padding. Use grayscale (or the watermark-suppressed channel from
  `preprocess.py`). Resize to a fixed square of 32-48 px, or letterbox like `form-field-v1-state` (40×160) if the
  label strip is included. Keep the aspect ratio. For a pure checkbox crop, a square of 32-48 is enough **(est.)**.
- **Architecture.** A tiny custom CNN (3-4 conv blocks of 16-64 channels, GAP, FC; 50-100k params) matches
  `form-field-v1-state` (96k params, depthwise-separable) [21b]. MobileNetV3-Small (2.5M) or ResNet18 (11.7M) with
  ImageNet weights only help when there are fewer than ~300 real crops **and** no synthetic data. Their ONNX files are
  10-45 MB, and they cost 2-10 ms/crop on CPU **(est.)**.
- **Classes.** `unchecked`, `checked`, plus either an explicit `ambiguous` class (strike-through, scribble,
  half-mark, watermark over the box) or a softmax confidence band (for example 0.35-0.65 → ambiguous). The band
  needs no extra labels, so start with that. Optionally add `not_checkbox` if the locator is the OpenCV candidate
  stage.
- **Data volume (est.).** 300-500 real crops per class from the samples plus 5-20k synthetic crops (rendered blank
  1004 boxes + drawn ✓/✗/fill + scan noise and red/blue overlays). The `form-field-v1-state` card shows this
  synthetic-only recipe reaches 0.981 on its split [21b]. Hold out real pages.
- **Training on CPU.** About 20k crops at 32-48 px with a 100k-param net take a few minutes per epoch on a laptop
  CPU, and minutes total for convergence **(est.)**. No GPU needed. Use `torch` only in a dev dependency group,
  never in the image.
- **Export / inference.** Export with `torch.onnx.export(..., dynamic_axes={"x": {0: "n"}})` and batch all crops of
  a page in one `session.run`. **(probe)** `form-field-v1-state` ran 30-94 crops in 7-21 ms at 1 thread (about
  0.2 ms/crop), and the model is 0.38 MB.
- **Public crop datasets.** `wendys-llc/chkbx` / checkbox-classifier (UI checkboxes, Apache-2.0), the harjot crop set,
  Roboflow sets (already listed in `deep-learning-and-data.md` §3). New here: `form-field-v1-state` can be used as
  a **pre-labeler/teacher** for our crops. Its `state_cnn.pt` (Apache-2.0) can also be fine-tuned directly [21b].

## 6. Stage 3: OCR of the label text

- **Crop strategy (recommended).** Do not OCR the full page on scans. **(probe)** RapidOCR at 1 thread on a 2550×3301
  page took about 10 s with 2.5 GB peak RSS, and det-only took 0.74 s with 1.5 GB. Instead, take a strip to the
  right of each box (height ≈ 1.5× box side, width ≈ 10-15× side, stopped at the next box or a vertical rule). Run
  **recognition only** (`eng(crop, use_det=False, use_cls=False)`): about 9 ms/crop, so 100 boxes ≈ 0.9 s at 258 MB
  peak **(probe)**. Recognition expects a single text line. Strips that span two labels produce merged text
  ("Fee SimpleLeasehold"), so cut strips at the next detected box.
- **Caveat.** Running RapidOCR with det on a small crop is slow (about 430 ms/crop in the probe), because the det
  stage upscales small inputs (`min_side_len`). Use recognition-only, or det on a mid-sized region at native scale.
- **Thread settings.** Set `EngineConfig.onnxruntime.intra_op_num_threads: 1` on Railway. RapidOCR's default is
  `enable_cpu_mem_arena: false` [12].
- **Born-digital PDFs: skip OCR.** pypdfium2 5.14 exposes `PdfTextPage.get_text_bounded(l,b,r,t)`,
  `get_charbox(i)`, `count_rects()/get_rect(i)` and `search()` [11]. **Verified:** on a test PDF,
  `get_text_bounded` next to a box returned `"Owner "`. Coordinates are PDF points with the origin at bottom-left.
  Convert with `px = pt × dpi/72` and `y_px = (page_h − y) × dpi/72`. A scanned PDF has no text layer:
  `count_chars()==0` means fall back to OCR.
- **AcroForm checkbox values with pypdfium2 (permissive). Verified** on a generated fillable PDF:
  ```python
  pdf.init_forms(); form = pdf.formenv.raw; page = pdf[i]
  for k in range(raw.FPDFPage_GetAnnotCount(page)):
      a = raw.FPDFPage_GetAnnot(page, k)
      if raw.FPDFAnnot_GetFormFieldType(form, a) == raw.FPDF_FORMFIELD_CHECKBOX:  # also FPDF_FORMFIELD_RADIOBUTTON
          checked = bool(raw.FPDFAnnot_IsChecked(form, a))
          r = raw.FS_RECTF(); raw.FPDFAnnot_GetRect(a, r)        # points, bottom-left origin
          # name: FPDFAnnot_GetFormFieldName(form, a, buf, n) with a c_ushort (UTF-16LE) buffer
      raw.FPDFPage_CloseAnnot(a)
  ```
  Output: `occ_owner True (80,698,92,710)` and `occ_tenant False (...)`. Run this under the existing global pdfium
  lock. **PyMuPDF is AGPL-3.0 or Artifex commercial** [16] (already noted in `pdf-rasterization.md`), so do not
  use it.

## 7. Several ONNX models in one FastAPI process (1 vCPU, 1-2 GB)

**Measured peak RSS (probe, macOS M4, ORT 1.31, 1 thread; Linux RSS is similar or slightly lower, est.):**

| What | After load | After 1 inference |
|---|---|---|
| Python + numpy + onnxruntime + Pillow + cv2 imports | 71 MB | — |
| Docling heron ONNX (640) | 411 MB | 590-630 MB |
| FFDNet-S ONNX (1216) | 138 MB | 670-710 MB |
| FFDNet-L ONNX (1216) | 211 MB | 1.07-1.14 GB |
| nano + state + wendys YOLO12n, all loaded, 4 pages processed (incl. 2550×4200 RGB arrays) | — | 720 MB total |
| RapidOCR rec-only, 100 crops | — | 258 MB total |
| RapidOCR full page (det+rec) | — | 2.5 GB total |

**Guidance:**
- Create each `InferenceSession` once in the FastAPI lifespan and share it. `session.run` is thread-safe [24].
  Set `intra_op_num_threads=1` and `inter_op_num_threads=1` to match 1 vCPU. Otherwise ORT spawns a thread per
  visible core and oversubscribes [24].
- Activation memory, not weights, dominates at high input sizes (FFDNet-S 38 MB weights → about 0.7 GB at 1216²).
  Prefer small inputs and tiles. Consider `enable_cpu_mem_arena=False` for big one-off shapes: it made little
  difference in the probe (±5%). Process pages sequentially, and keep the existing PDF page cap.
- Plan for nano (tiled) + state + RapidOCR-rec + the 300-DPI page array ≈ **0.4-0.6 GB peak (est.)**, which fits 1 GB.
  Adding heron or FFDNet-L would not fit safely in 1 GB.
- **Image size.** `onnxruntime` 1.31.0 manylinux wheel is 23.8 MB (75 MB installed on macOS). `rapidocr` is 27 MB with
  models. Models: nano 3.7 MB, state 0.4 MB. **torch** from PyPI is a 555 MB wheel and pulls CUDA wheels
  (several GB, est.). The CPU-index wheel is about 190 MB, roughly 0.7-1 GB installed **(est.)**. Keep torch out of the
  runtime image. `rapidocr` depends on `opencv-python-headless`, the same package we already ship, so no duplicate cv2.

---

## Sources

1. Docling heron model card + config/preprocessor: https://huggingface.co/docling-project/docling-layout-heron (config: https://huggingface.co/docling-project/docling-layout-heron/blob/main/config.json , https://huggingface.co/docling-project/docling-layout-heron/blob/main/preprocessor_config.json)
2. Docling heron ONNX: https://huggingface.co/docling-project/docling-layout-heron-onnx
3. Docling egret-medium card: https://huggingface.co/docling-project/docling-layout-egret-medium
4. Livathinos et al., "Advanced Layout Analysis Models for Docling" (arXiv 2509.11720): https://arxiv.org/abs/2509.11720
5. FFDNet-L card (no license field): https://huggingface.co/jbarrow/FFDNet-L
6. FFDNet-L ONNX: https://huggingface.co/jbarrow/FFDNet-L-cpu
7. FFDNet-S ONNX: https://huggingface.co/jbarrow/FFDNet-S-cpu
8. commonforms repo (inference.py, pyproject; no LICENSE): https://github.com/jbarrow/commonforms ; PyPI: https://pypi.org/project/commonforms/
9. FFDetr (Apache-2.0): https://huggingface.co/jbarrow/FFDetr
10. docling-ibm-models pyproject (MIT, torch deps, opencv<5): https://github.com/docling-project/docling-ibm-models/blob/main/pyproject.toml ; docling LICENSE (MIT): https://github.com/docling-project/docling/blob/main/LICENSE
11. pypdfium2 API docs (PdfTextPage, raw FPDFAnnot_*): https://pypdfium2.readthedocs.io/en/stable/python_api.html ; PDFium `fpdf_annot.h` (FPDFAnnot_IsChecked, GetFormFieldType): https://pdfium.googlesource.com/pdfium/+/refs/heads/main/public/fpdf_annot.h
12. RapidOCR docs (Apache-2.0; models © Baidu): https://rapidai.github.io/RapidOCRDocs/main/ ; repo: https://github.com/RapidAI/RapidOCR
13. rapidocr on PyPI (3.10.0): https://pypi.org/project/rapidocr/
14. PaddleOCR 3.x (Apache-2.0) / paddlex deps: https://pypi.org/project/paddleocr/ , https://pypi.org/project/paddlex/ , https://pypi.org/project/paddlepaddle/
15. PP-OCRv6 introduction: https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html
16. PyMuPDF license (AGPL / commercial): https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright
17. docTR (python-doctr 1.1.0, torch deps): https://pypi.org/project/python-doctr/ ; https://github.com/mindee/doctr
18. OnnxTR README (Apache-2.0, CPU benchmarks vs docTR/Tesseract/PaddleOCR): https://github.com/felixdittrich92/OnnxTR
19. tesserocr (MIT; tessdata separate): https://github.com/sirfz/tesserocr ; https://pypi.org/project/tesserocr/
20. form-field-v1-nano (Apache-2.0, YOLOX-Nano ONNX): https://huggingface.co/nutrientdocs/form-field-v1-nano
21. form-field-v1-small (commercial; baselines incl. FFDetr): https://huggingface.co/nutrientdocs/form-field-v1-small
    21b. form-field-v1-state (Apache-2.0): https://huggingface.co/nutrientdocs/form-field-v1-state
    21c. form-field-v1-benchmark: https://huggingface.co/datasets/nutrientdocs/form-field-v1-benchmark
22. wendys-llc/checkbox-detector (YOLO12n ONNX): https://huggingface.co/wendys-llc/checkbox-detector
23. pdfme form-field detector: https://huggingface.co/takumi123xxx/pdfme-form-field-detector
24. ONNX Runtime threading / session options: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
25. onnxruntime / torch wheel sizes: https://pypi.org/project/onnxruntime/#files , https://pypi.org/project/torch/#files , https://download.pytorch.org/whl/cpu/torch/
26. YOLOX (Apache-2.0): https://github.com/Megvii-BaseDetection/YOLOX
27. Ultralytics licensing (trained models AGPL): https://www.ultralytics.com/license
