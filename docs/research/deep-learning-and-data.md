# Deep-learning options and training data for checkbox detection

Researched 2026-10-07. Scope: `POST /detect` takes a page image of a Fannie Mae 1004 / 1004MC / 1004C form and returns
`{"boxes":[{"bbox":[x1,y1,x2,y2],"is_checked":bool}]}`. Constraints: 4 unlabeled samples, boxes of about 10-20 px,
noise (grid lines, blue shading, red diagonal watermark, handwriting), CPU inference in Docker.

Bracketed numbers like [3] point to the Sources section. Claims marked **(est.)** are my own estimates and were not
measured or published.

---

## TL;DR

- **No off-the-shelf model does the whole job.** No public pretrained model both localizes checkboxes on scanned
  forms *and* labels them checked/unchecked with a permissive license. The closest is **FFDNet (CommonForms)**
  [14][15]. It detects "ChoiceButton" (checkbox/radio) on form pages but does not give the checked state, and it is
  built on Ultralytics (AGPL).
- **Recommended path: hybrid.** Use OpenCV to find candidate squares (high recall), then a tiny CNN to decide
  checkbox vs. not and checked vs. unchecked. The data need is small: crops are cheap to label, and synthetic crops are
  easy to make. It runs in milliseconds on CPU and has no license risk (PyTorch/ONNX Runtime).
- **Stretch: fine-tune a detector.** RF-DETR-N/S (Apache-2.0) or D-FINE-N/S (Apache-2.0) are the license-clean choices.
  YOLO26 is the most convenient and the fastest on CPU, with small-object improvements (STAL), but it is AGPL-3.0, so
  weights trained with it are AGPL too [2][3].
- **Avoid DEIMv2 for commercial use.** Its license changed to non-commercial only [9].
- **VLMs (Qwen3-VL/3.5, Gemini, Florence-2, PaliGemma 2) are not a fit for the serving path.** They are slow on CPU,
  their coordinates are quantized (0-1000 grid), and they are unreliable on 10-px targets and dense pages. They are
  useful offline for pre-labeling or checking the state of a crop.
- **Data plan:** pseudo-label the 4 samples with the OpenCV pipeline, correct them in Label Studio/CVAT (about 150-300
  boxes per page), and generate synthetic forms and crops. Hold out real pages for evaluation and never train on them.

---

## 1. Summary comparison

| Option | Pros | Cons | License | Data needed | Effort | CPU latency / page (est.) |
|---|---|---|---|---|---|---|
| **A. Hybrid: OpenCV candidates + tiny CNN** | Fast, explainable, little data, robust to the watermark if the classifier sees it in training | Recall is capped by the candidate stage; struggles when handwriting or grid lines break the box contour | Yours (PyTorch BSD, OpenCV Apache-2.0) | ~1-3k crops (real + synthetic) | Low-Med | 50-200 ms (OpenCV dominates; CNN on 300 crops of 32x32 takes under 10 ms) |
| **A'. Hybrid with pretrained classifier** (EfficientNet-B0 / MobileNetV3 fine-tune) | ImageNet features, few samples | Overkill for 32-px crops; 4M params | Apache-2.0 (timm/torchvision weights) | ~500-1k crops | Low | +20-80 ms for 300 crops at 64-96 px |
| **B. YOLO26 (Ultralytics) fine-tune, 2 classes** | Best tooling, NMS-free, STAL helps small targets, 1-line ONNX export, fastest on CPU | **AGPL-3.0 covers trained weights**; needs hundreds of labeled pages for robustness; needs high resolution (≥1280) or tiling | AGPL-3.0 / Enterprise [2][3] | 50-300 pages labeled or synthetic | Med | n@640 is 39 ms [1]; a full page at about 1600 px is about 200-400 ms (est.) |
| **C. RF-DETR-N/S fine-tune** | Apache-2.0, DINOv2 backbone, strong transfer with small datasets (RF100-VL), ONNX export | Native input 384-512 px is too small for 12-px boxes, so tiling is required; DETR-style models are slower on CPU; no official CPU figures | Apache-2.0 (N-L); PML for others [5][6] | 50-300 pages | Med | 0.3-1.5 s with tiling (est.) |
| **D. D-FINE-N/S fine-tune** | Apache-2.0, small (N about 4M params), ONNX export scripts | Research-grade training repo, COCO-format config | Apache-2.0 [8] | 50-300 pages | Med-High | Similar to C (est.) |
| **E. FFDNet (CommonForms) zero-shot + state classifier** | Already trained on 450k+ form pages including checkboxes, runs on CPU | Trained on born-digital fillable PDFs, not noisy scans; finds empty widget locations, no checked state; Ultralytics-based (AGPL) | Dataset Apache-2.0; code/model license unstated; depends on `ultralytics` [14][15][16] | 0 for localization, crops for state | Low | ~1-3 s at 1216-1600 px (est.) |
| **F. VLM grounding** (Qwen3-VL/3.5, Gemini, Florence-2, PaliGemma 2) | Zero-shot, can read the label next to the box | Slow (seconds to tens of seconds on CPU), misses or merges tiny boxes, coordinates on a 0-1000 grid, hallucinations; Gemini sends data out | Apache-2.0 / MIT / Gemma terms / proprietary | 0 (or about 1k for fine-tuning) | Low (API) to High (fine-tune) | 5-60 s CPU for 2-8B models (est.) |
| **G. Train detector from scratch** | Fully custom | Pointless with this little data; pretrained backbones always win | Yours | Thousands of pages | High | n/a |

**Pick:** A as the shipped baseline. Put B or C behind the same interface as an experiment if time allows. Use F only
offline as a labeling aid.

---

## 2. Object detectors (state as of Oct 2026)

### 2.1 Ultralytics YOLO26 (current), YOLO11 (previous)
- **Current version.** YOLO26 is the newest released Ultralytics model (January 2026). YOLO27 is announced as "coming
  soon" [1]. The `ultralytics` package is at v8.4.174 (2026-10-06) [4].
- **Changes relevant here.** NMS-free end-to-end head, DFL removed (simpler export), ProgLoss, and **STAL
  (Small-Target-Aware Label Assignment)**, which improves positive-label coverage for small objects [1].
- **Speed.** COCO 640 px, CPU ONNX: YOLO26n 40.9 mAP at 38.9 ms; YOLO26s 48.6 mAP at 87.2 ms [1].
- **License.** AGPL-3.0 or a paid Enterprise license. Ultralytics states that "all Ultralytics YOLO trained models
  fall under AGPL-3.0" and that SaaS/API use needs either open-sourcing the whole project or an Enterprise license [2][3].
  For a take-home this is fine if the repo is public, but say so explicitly in the README.
- **Small datasets.** The best ergonomics of any option: a YOLO-format dataset, then
  `yolo train model=yolo26n.pt data=cb.yaml imgsz=1280`. Ultralytics' own guideline is ≥1,500 images and ≥10k instances
  per class for production quality, and to train at native or higher resolution when objects are small [10]. One form
  page already holds 100-300 checkbox instances, so instance counts climb quickly.
- **Sketch:**
  1. Render or label pages, then export YOLO txt with classes `0=unchecked, 1=checked`.
  2. Tile pages into 1024x1024 crops with 20% overlap, or train at `imgsz=1280-1600`.
  3. Fine-tune `yolo26n.pt` with augmentation (hsv, small rotate, blur), `mosaic` on, and `fliplr=0` (an X is
     symmetric, but text is not).
  4. `model.export(format="onnx")`, then use `onnxruntime` in the API and merge tiles with NMS when tiling.

### 2.2 RF-DETR (Roboflow)
- **Status.** Active; package `rfdetr` at 1.11.2 (2026-10-04) [7]. Paper arXiv:2511.09554, ICLR 2026 [5].
- **Sizes and licenses.** Nano 384 px (48.4 AP), Small 512 px (53.0 AP), Medium 576 px, Large 704 px, all
  **Apache-2.0**. Atto/Femto/Pico/XL/2XL are under the Platform Model License (PML 1.0) and need `rfdetr[plus]` [5].
- **Export.** `model.export(format="onnx")` runs on CPU and produces a single `.onnx` file. It accepts COCO JSON or
  YOLO format datasets [6].
- **CPU.** The docs only say "CPU inference is supported for evaluation". All published latencies are T4/TensorRT [6].
- **Fit.** Good transfer from a DINOv2 backbone to small custom datasets, which is the stated goal of RF100-VL.
  The low native resolution is a problem for 10-20 px boxes on a 2550x3300 scan, so **tiling is mandatory**.

### 2.3 D-FINE / DEIM / DEIMv2
- **D-FINE.** Apache-2.0, actively maintained (pushed 2026-10) [8]. It is the base detector that DEIM builds on.
  Has ONNX/TensorRT export tooling.
- **DEIM (v1).** Apache-2.0 [9b]. CVPR 2025.
- **DEIMv2.** DINOv3 features, sizes from Atto (0.5M) to X [9]. **The current LICENSE.md is non-commercial only:**
  "No rights are granted for Commercial Use", which covers SaaS/APIs [9]. Do not use it.
- **Training format.** COCO JSON with `remap_mscoco_category: False` and a `custom_detection.yml` [9].

### 2.4 RT-DETR / RT-DETRv2
- Apache-2.0 (lyuwenyu/RT-DETR) [11]. RT-DETRv2 is in HF Transformers as `RTDetrV2` [12]. The repo supports
  onnxruntime/OpenVINO deployment [11]. It has been superseded on accuracy by D-FINE and RF-DETR. The Ultralytics
  wrapper of RT-DETR is AGPL; the upstream repo is not.

### 2.5 YOLOX
- Apache-2.0, but last pushed 2025-06 [13]. Its main appeal today is a permissive YOLO, but it is effectively
  unmaintained and inferior to D-FINE/RF-DETR. Not recommended.

### 2.6 Small-object notes (applies to all detectors)
- Checkboxes are about 0.4% of page width. At 640 px input a 12-px box shrinks to about 3 px, below what any stride-8
  head can resolve. Use **tiling** (SAHI-style) or **native resolution**, and train and infer at the same scale [10].
- Use few classes. Two (`checked`, `unchecked`) is better than one class plus a separate classifier only if there is
  enough checked data. Otherwise detect `checkbox`, then classify (detector plus stage 2 of option A).

---

## 3. Already-pretrained checkbox resources

| Resource | What it is | Classes | License | Notes |
|---|---|---|---|---|
| **FFDNet-S / FFDNet-L** (jbarrow) [14][15] | 25M-param form-field detector (L), `.pt` (Ultralytics) | TextBox, **ChoiceButton**, Signature | Model card: none stated; dataset Apache-2.0; package depends on `ultralytics` and `rfdetr` [16] | ChoiceButton AP 78.1 (L, 1216 px). Trained on fillable born-digital PDFs, so it gives **no checked state**. The best zero-shot localizer to try on our scans. |
| **CommonForms dataset** [14][17] | ~55k docs / 450k+ pages from Common Crawl fillable PDFs | same 3 | **Apache-2.0** (HF card) | Largest checkbox-location dataset. Labels come from PDF widgets and are noisy; "Mind the Gaps" (arXiv 2609.23679) adds a curated mini-CommonForms benchmark [18]. |
| **wendys-llc/checkbox-classifier** [19] | EfficientNet-B0, checked/unchecked, 224 px crops | 2 | Apache-2.0 | Trained on ~4.8k **UI screenshot** checkboxes, so expect a domain gap with scans. Useful as a init/baseline. |
| **linhdo/checkbox-detector** (HF Space) [20] | 2-stage YOLO: detector + classifier (block/checked/unchecked) | 3 | Ultralytics (AGPL) | Demonstrates the two-stage design. |
| **harjotsingh2004 checkbox dataset / PaliGemma FT** [21] | 1,000 crops at 224 px, checked/unchecked/partial | 3 | Apache-2.0 (claimed) | Unverified repo; small crops, not full pages. |
| **Roboflow Universe** sets [22] | e.g. `checkbox-detector` (298 imgs, checked/unchecked), `checkbox-0fyo0` (165 imgs), `check_box` (107 imgs), `checkbox_v2` (CC BY 4.0) | varied names | Mostly CC BY 4.0, check per project | Small, inconsistent class names, mixed domains (forms, UI). Fine as extra pretraining data after remapping classes. |
| **CheckboxQA** (Snowflake) [23][24] | ~600 QA pairs over multi-page docs with checkboxes | QA, **no boxes** | **CC BY-NC**, eval only | A VLM benchmark, not training data for a detector. |

**Document-layout models: none detects checkboxes as a class.**
- **DocLayout-YOLO** (YOLOv10/Ultralytics, AGPL-3.0) [25]. Its DocStructBench classes are layout regions (title,
  text, figure, table, formula, captions...).
- **LayoutLMv3** [26]. Token classification over OCR words, not a detector. Weights are CC BY-NC-SA 4.0.
- **Table Transformer**. Table and row/column structure only. Could help mask table grid lines, but that is not worth it.
- **Florence-2** `<OD>`/grounding [27]. Generic classes; checkboxes are not reliably produced zero-shot (see section 6).

---

## 4. Option A in detail: classical candidates + small CNN (recommended)

**Why it fits.** Checkboxes on these forms are printed, axis-aligned, near-square, and of one or two sizes per
template. Geometry finds them; the hard part is the *state* under noise (watermark, handwriting, shading). That is a
32x32 classification problem, which needs very little data.

**Pipeline sketch:**
1. Preprocess: grayscale. To suppress the red watermark and blue shading, use the channel where they are faint, or
   threshold in HSV and remove the hue. Then use adaptive threshold (`cv2.adaptiveThreshold`).
2. Get candidates by two complementary methods for recall:
   - Contours: `findContours(RETR_TREE)` → `approxPolyDP` with 4 vertices, aspect 0.8-1.25, side length 8-30 px
     (scaled by DPI), solidity check.
   - Morphology: open with short horizontal and vertical kernels → line map → small closed squares. This also catches
     boxes whose border is broken by an X.
3. Dedupe with NMS (IoU > 0.3) and drop candidates that sit inside table cells (cell size ≫ box size).
4. Crop each candidate with about 25% padding and resize to 32x32 (or 48x48).
5. Classify each crop with a CNN into 3 classes: `not_checkbox`, `unchecked`, `checked`. The `not_checkbox` class
   cleans up false positives from step 2 (letters such as "O" or "D", table cells, digits).
6. Return boxes with `is_checked = (class == checked)`. Optionally expose the score.

**Model options:**
- **Tiny custom CNN.** 3 conv blocks (16-32-64 channels), GAP, FC, about 50k params. Trains in minutes on CPU and
  exports to ONNX with `torch.onnx.export`.
- **MobileNetV3-Small / EfficientNet-B0 transfer** (torchvision/timm weights, Apache-2.0/BSD). Upsample crops to
  64-96 px. Better if the real crop data stays under ~300. Can start from wendys-llc's EfficientNet-B0 checkbox
  classifier [19].
- **Non-DL baseline for comparison:** fill ratio of dark pixels inside the inner 60% of the box, with a threshold.
  This is the "reasoning about tradeoffs" story: show where the CNN beats it (watermark, faint ✓, handwriting crossing
  the box).

**Data needs (est.).** About 300-500 real crops per class from the 4 pages (pseudo-labeled, then corrected), plus
5-20k synthetic crops. Augment with JPEG quality 30-90, Gaussian blur σ ≤ 1.2, ±3° rotation, ±2 px shift, red/blue
overlays, random pen strokes, and partial borders.

**Tradeoff.** The system can never find a box that the candidate stage misses. Measure candidate recall separately
(section 7).

---

## 5. Option B/C/D in detail: fine-tuned detector

**When it's worth it.** When the candidate stage's recall plateaus (handwriting over boxes, broken borders, varied
DPI) and a few hundred labeled pages are available, real or synthetic.

**Sketch (framework-agnostic):**
1. Build the dataset (section 8): synthetic pages, pseudo-labeled real pages, and CommonForms pages for a `checkbox`
   prior.
2. Tile to 1024 px with 20% overlap. Drop most empty tiles (keep 0-10% background tiles [10]).
3. Choose classes: `checked` / `unchecked`, or `checkbox` plus a crop classifier.
4. Fine-tune from COCO weights for 50-150 epochs, early-stopped on mAP50 on a held-out real page.
5. Export to ONNX. At inference, tile → run → shift boxes back → global NMS → JSON.
6. Calibrate the score threshold on the eval set, targeting the F1-optimal point.

**Which one.**
- Public repo and speed matter most: YOLO26n/s (AGPL acceptable).
- License-clean: RF-DETR-N/S, or D-FINE-N/S if a smaller model is needed.
- Zero-shot localization prior: FFDNet-L, then the option A classifier for state. Try this first; it costs one hour.

---

## 6. VLM detection / grounding

| Model | Size | License | Box format | Notes |
|---|---|---|---|---|
| **Qwen3-VL** (2B/4B/8B/32B, MoE 30B-A3B, 235B) [28] | 2-235B | Apache-2.0 | `bbox_2d`, relative coords on a 0-1000 grid (Qwen2.5-VL used absolute pixels) [28][29] | Released Sep-Oct 2025. Supports 2D box/point grounding. |
| **Qwen3.5** (0.8B ... 397B-A17B) [30] | 0.8-397B | Apache-2.0 | 0-1000 grid | Released Feb 2026. Grounding IoU@0.5 is close to Qwen3-VL in third-party benchmarks [31]. |
| **Gemini** (API) [32] | n/a | Proprietary | `box_2d` = `[ymin,xmin,ymax,xmax]` on 0-1000 | Higher media resolution helps small details. Larger images are tiled at 768x768 [32]. Sends documents to Google, a PII concern for appraisals. |
| **Florence-2** base/large [27][33] | 0.23B / 0.77B | MIT | `<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>` → pixel boxes | Fixed **768x768** input [33], so a 12-px box on a 3300-px page becomes about 3 px. Only usable on tiles or after fine-tuning. |
| **PaliGemma 2** 3B (224/448/896) [34] | 3B+ | Gemma terms | `<loc>` tokens | Base model, needs fine-tuning for detection [34]. |

**Assessment.**
- **Tiny objects.** A 0-1000 grid on a 2550-px page quantizes coordinates to about 2.5 px, which is 15-25% of a box
  side, so IoU suffers. Dense pages with 200+ boxes exceed what generative decoders list reliably: they drop or
  duplicate items (est., consistent with CheckboxQA findings that LVLMs struggle with checkable content [23]).
- **Latency.** Seconds per page on GPU; tens of seconds to minutes on CPU for 2-8B models (est.). Not acceptable for
  a CPU Docker service.
- **Good uses.** (a) Offline pre-labeling of checked state on crops ("Is this box checked? yes/no"). (b) A second
  opinion during label QA. (c) Optional `label` text near each box (out of scope).

---

## 7. Evaluation

**Metrics:**
- **Localization:** precision, recall, and F1 at IoU ≥ 0.5, matched greedily by IoU, ignoring state. Also report
  IoU ≥ 0.3, because 1-2 px errors on 12-px boxes swing IoU a lot.
- **State:** accuracy (and a confusion matrix) on matched pairs only, so detection and classification errors stay separate.
- **End-to-end:** per-class P/R/F1 where a TP needs IoU ≥ 0.5 *and* the correct state. This is the number to put
  in the README.
- **mAP50 / mAP50-95 (COCO-style)** for comparing detectors. mAP50-95 penalizes tiny boxes heavily, so treat it as
  secondary.
- **Candidate recall** (option A only): the share of GT boxes covered by any candidate. This is the ceiling for the
  whole system.
- **Latency:** p50/p95 per page in the Docker container on CPU.

**Building the eval set from 4 images:**
1. Run the OpenCV pipeline and export candidates to Label Studio/CVAT as pre-annotations.
2. Correct them by hand: add missed boxes, delete false positives, set the `checked` attribute. Budget about 20-40 min
   per page.
3. Two-pass QA: a second look at all `checked` boxes and every box touched by handwriting or the watermark.
4. Freeze it in COCO JSON (or a simple `{"boxes":[...]}` per image, matching the API). **Never train on these pages.**
   If they are needed for training, do leave-one-page-out cross-validation.
5. Tag hard cases (watermark overlap, handwriting crossing, partial fill, ✓ vs X) for per-slice metrics.
6. Script: `eval.py gt/ pred/` prints P/R/F1@0.5, state accuracy, and the confusion matrix. Use `pycocotools` or
   `supervision` for mAP.

---

## 8. Training-data acquisition

### 8.1 Public data
- **CommonForms** (Apache-2.0, 450k+ pages, ChoiceButton class) [14][17]. Best source of checkbox *locations* on real
  form layouts. No checked state, and born-digital rather than scanned, so add scan augmentation.
- **Roboflow Universe** checked/unchecked sets (100-300 images each, mostly CC BY 4.0) [22]. Remap class names; mixed domain.
- **wendys-llc/chkbx** (~6k UI checkbox crops) [19]. Useful only for classifier pretraining.
- **FUNSD / XFUND**. Labels are question/answer/header/other entities with no checkbox class. Their scanned forms do
  contain checkboxes, so they are a source of unlabeled realistic noise for pseudo-labeling.
- **CheckboxQA** is CC BY-NC and eval-only, with no boxes [24]. Skip it for training.
- **The forms themselves.** Fannie Mae publishes blank 1004/1004MC/1004C PDFs. Rasterizing them gives perfect
  templates with **exact checkbox coordinates**, either from PDF vector drawings/widgets or from our OpenCV pipeline
  on clean renders. This is the highest-value source.

### 8.2 Synthetic generation (highest leverage)
1. Rasterize the blank official forms at 150/200/300 DPI with `pdftoppm`/PyMuPDF. Extract checkbox rects from the PDF
   (widgets or vector rects), or run the detector on the clean render and verify once by hand.
2. For each box, sample a state: unchecked (50%), X (30%), ✓ (10%), filled/scribble (5%), partial or off-center
   mark (5%). Draw marks with random stroke width, slight overshoot, and jitter. Use handwriting-like fonts or
   Bezier strokes, in black or blue ink.
3. Fill text fields with fake typed or handwritten text, some of which crosses boxes.
4. Overlay a red diagonal "SAMPLE"/watermark with random alpha and angle, plus blue cell shading.
5. Apply scan degradations: rotation ±1.5°, perspective jitter, Gaussian/salt noise, blur, JPEG q 30-95, downscale and
   upscale, contrast/gamma shifts, and binarization artifacts (`albumentations` or `augraphy`).
6. Emit YOLO/COCO labels for the detector and crops for the classifier from the same generator.
- Use 500-2,000 synthetic pages; at about 150 boxes each that gives 75k-300k instances (est.). Domain-check by
  evaluating only on the real held-out pages.

### 8.3 Pseudo-labeling loop
OpenCV candidates → current classifier → export to Label Studio or CVAT with pre-annotations → human correction →
retrain → repeat. Correct the low-confidence predictions first (active learning). Two or three rounds over the 4 real
pages plus FUNSD-like scans is usually enough for a take-home (est.).

### 8.4 Annotation tools
- **Label Studio** (Apache-2.0): ML-backend pre-annotation, rectangle plus choice attribute, exports YOLO/COCO.
- **CVAT** (MIT): fast box editing, attributes, auto-annotation with ONNX models, exports YOLO/COCO.
- **Roboflow** (SaaS): annotation, augmentation, and one-click RF-DETR training. Free tier datasets are public, so watch for PII.

### 8.5 How much labeled data?
- **Ultralytics guideline for production:** ≥1,500 images and ≥10k instances per class, 0-10% background, and every
  instance labeled ("partial labeling will not work") [10].
- **In practice for fine-tuning from COCO/DINOv2 weights (est.):** 50-100 well-labeled pages (5k-20k boxes) of one
  form family gives usable mAP50, and synthetic pages close most of the rest of the gap. Option A's classifier
  needs only a few hundred real crops per class with synthetic augmentation.
- **Label quality matters more than quantity** for small objects. One missing box per page teaches the model to
  ignore that pattern.

---

## 9. Recommendation for the take-home

1. Ship **option A** (OpenCV candidates + 3-class tiny CNN in ONNX), trained on synthetic crops from rendered blank
   1004 forms plus corrected pseudo-labels from 2-3 of the samples. Hold out at least 1 real page for eval.
2. Run **FFDNet-L zero-shot** once as a localization comparison, and report its recall versus OpenCV.
3. If time remains, fine-tune **RF-DETR-N** (Apache) or **YOLO26n** (AGPL, document it) on synthetic tiles. Compare
   end-to-end F1 and CPU latency in a table in the README.
4. Document what was not done: VLMs (latency, quantization, PII) and from-scratch training (no data), and why.

---

## Sources

1. Ultralytics YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
2. Ultralytics licensing: https://www.ultralytics.com/license
3. Ultralytics repo (AGPL-3.0): https://github.com/ultralytics/ultralytics
4. Ultralytics releases (v8.4.174, 2026-10-06): https://github.com/ultralytics/ultralytics/releases
5. RF-DETR repo (model table, licenses): https://github.com/roboflow/rf-detr
6. RF-DETR docs (ONNX export, dataset formats, CPU note): https://rfdetr.roboflow.com/latest/
7. RF-DETR releases (1.11.2, 2026-10-04): https://github.com/roboflow/rf-detr/releases
8. D-FINE repo (Apache-2.0): https://github.com/Peterande/D-FINE
9. DEIMv2 repo and LICENSE.md (non-commercial): https://github.com/Intellindust-AI-Lab/DEIMv2 ; paper https://arxiv.org/abs/2509.20787
   9b. DEIM v1 (Apache-2.0): https://github.com/Intellindust-AI-Lab/DEIM
10. Ultralytics tips for best training results: https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/
11. RT-DETR repo (Apache-2.0): https://github.com/lyuwenyu/RT-DETR
12. HF Transformers RT-DETRv2: https://huggingface.co/docs/transformers/model_doc/rt_detr_v2
13. YOLOX repo: https://github.com/Megvii-BaseDetection/YOLOX
14. CommonForms paper: https://arxiv.org/abs/2509.16506
15. FFDNet-L model card: https://huggingface.co/jbarrow/FFDNet-L
16. commonforms package: https://github.com/jbarrow/commonforms
17. CommonForms dataset: https://huggingface.co/datasets/jbarrow/CommonForms
18. Mind the Gaps (mini-CommonForms): https://arxiv.org/abs/2609.23679
19. wendys-llc checkbox classifier: https://huggingface.co/wendys-llc/checkbox-classifier
20. linhdo checkbox-detector Space: https://huggingface.co/spaces/linhdo/checkbox-detector
21. PaliGemma checkbox FT / dataset page: https://huggingface.co/harjot997u7/test
22. Roboflow Universe checkbox sets: https://universe.roboflow.com/ujiboo/checkbox-detector-zxwoh , https://universe.roboflow.com/tempqqaq/checkbox_v2 , https://universe.roboflow.com/talha-4wrwc/checkbox-0fyo0 , https://universe.roboflow.com/vidyashree/check_box-ohkys
23. CheckboxQA paper: https://arxiv.org/abs/2504.10419
24. CheckboxQA repo (CC BY-NC): https://github.com/Snowflake-Labs/CheckboxQA
25. DocLayout-YOLO (AGPL-3.0): https://github.com/opendatalab/DocLayout-YOLO
26. LayoutLMv3 (CC BY-NC-SA 4.0): https://huggingface.co/microsoft/layoutlmv3-base
27. Florence-2 base model card (MIT, tasks): https://huggingface.co/microsoft/Florence-2-base
28. Qwen3-VL repo: https://github.com/QwenLM/Qwen3-VL
29. Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
30. Qwen3.5 release coverage: https://www.deeplearning.ai/the-batch/alibabas-latest-flagship-models-are-open-weights-moe-performers-in-sizes-from-less-than-1b-parameters
31. Qwen3.5 grounding, third-party benchmark: https://arxiv.org/abs/2606.00828
32. Gemini image understanding (box_2d format): https://ai.google.dev/gemini-api/docs/image-understanding
33. Florence-2 preprocessor config (768x768): https://huggingface.co/microsoft/Florence-2-large/blob/main/preprocessor_config.json
34. PaliGemma 2 896 model card: https://huggingface.co/google/paligemma2-3b-pt-896
