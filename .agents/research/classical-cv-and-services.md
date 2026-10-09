# Checkbox detection: classical CV and managed services

Research date: 2026-10-07. Scope: non-deep-learning (OpenCV) pipelines, open-source checkbox libraries, managed document-AI APIs, and general vision-language models (VLMs), for `POST /detect` on Fannie Mae 1004/1004MC/1004C page images.

Legend: **[doc]** means the claim comes from a primary source listed under Sources. **[est]** means it is my engineering estimate and has not been measured on our 4 samples. Measure before quoting it.

## Summary table

| Option | Pros | Cons | Effort | Expected accuracy on these docs | CPU latency / page |
|---|---|---|---|---|---|
| **A. Custom OpenCV pipeline** (binarize, remove lines, contours, size/aspect/squareness filters, ink-ratio fill) | No network, no keys, deterministic, explainable, every threshold is tunable; ships as one small Python module; fits a take-home | Hand-tuned thresholds; breaks when boxes touch table lines, when scribbles cross boxes, or at scales it wasn't tuned for; no learned notion of "this is a checkbox" vs a small square glyph | 1–2 days incl. tuning on the 4 samples | Detection recall roughly 85–95% and fill accuracy roughly 90%+ on clean form regions; worse where boxes are inside grid cells or under scribbles [est] | ~50–300 ms for a ~2500×3300 px page [est] |
| **B. `boxdetect` (PyPI)** | Ready-made `get_checkboxes` pipeline built on the same ideas as A; MIT | Last release 1.0.2 on 2023-01-18 (unmaintained since); needs pixel `width_range`/`height_range` per document scale; fill check is a single global pixel ratio; pulls in scikit-learn and imutils | 0.5 day to integrate + tuning | Similar to A at best, since it is the same technique with less control [est] | Similar to A; grows with the number of `scaling_factors` [doc] |
| **C. Azure AI Document Intelligence** `prebuilt-layout` selection marks | Best-documented checkbox output: `state` selected/unselected, `polygon`, `confidence`; pixel units for images; data deleted after 24 h | Network + Azure account; ~US$10/1k pages [secondary source]; async polling; boxes come back as polygons you then convert; no control when it misses | 0.5 day | Likely strong on clean, printed boxes; unknown under red watermark and scribbles [est] | Not CPU-bound; round trip is typically seconds per page [est] |
| **D. AWS Textract** `AnalyzeDocument` FORMS/TABLES | `SELECTION_ELEMENT` with `SELECTED`/`NOT_SELECTED`, confidence, bbox [doc] | FORMS costs $50/1k pages, TABLES $15/1k [doc]; bbox normalized 0–1 (multiply by page size) [doc]; with `FeatureTypes=TABLES` you only get boxes inside tables [doc]; inputs may be used to improve AWS ML unless the org opts out [doc] | 0.5 day | Likely good; fails similarly to C [est] | Seconds per page [est] |
| **E. Google Document AI Form Parser** | Checkboxes returned as key-value pairs with filled/unfilled `valueType` [doc] | $30/1k pages [doc]; no radio buttons; some boxes come back without a key; parsing of unfilled values is weak [doc]; geometry is via normalized vertices | 0.5–1 day (GCP project, processor setup) | Medium; the docs themselves say blank values are a weak spot [doc] | Seconds per page [est] |
| **F. General VLM** (GPT, Gemini, Claude, Qwen2.5-VL) asked for bboxes | Zero tuning; can read context ("Yes/No" labels) | Coordinates are approximate [doc]; images are downscaled (Claude: long edge 1568–2576 px) [doc], so a 12 px box can shrink to ~7 px; the best model scored 83.2 on CheckboxQA vs 97.5 human, even with no localization required [doc]; cost and nondeterminism | 0.5 day | Poor for pixel-tight bboxes on ~50–100 tiny boxes per page; fine as a verifier on crops [est] | Seconds to tens of seconds per page [est] |

**Recommendation for the take-home:** build **A** as the shipped solution, borrowing ideas from `boxdetect` without depending on it. Put detection and fill classification behind two separate functions so a managed service or learned model can replace either one later. Write up C/D/E as the "production alternative / ground-truth generator" and F as "not suitable for bboxes, maybe for a second-pass verifier on ambiguous crops". Reasons: the reviewers can run A offline with no credentials, its tradeoffs are explicit and tunable, and with no labels there is no way to compare services fairly anyway.

---

## A. Custom OpenCV pipeline

### A1. Pre-processing / binarization
- Convert to grayscale. The **red watermark and blue shaded cells are mostly a color problem**: before graying, either take the channel where ink is darkest and colored overlays are lightest, or drop pixels with high HSV saturation (red and blue are saturated, black ink is not). This is cheaper and more reliable than trying to undo the effect after thresholding [est].
- Global **Otsu** picks the threshold that minimizes intra-class variance for a bimodal histogram [doc: OpenCV thresholding]. It works for clean renders, but shaded cells move the histogram.
- **Adaptive threshold** (`ADAPTIVE_THRESH_GAUSSIAN_C`, block ~31–51 px, C ~10–15) handles uneven backgrounds and shaded cells [doc: OpenCV thresholding]. Default to adaptive, or use Otsu after removing saturated pixels.
- Invert, so ink = 255. Optionally apply a 3×3 median filter first to remove scan speckle.

### A2. Line extraction (needed because forms are tables)
- OpenCV's documented recipe uses `MORPH_OPEN` with long thin structuring elements, e.g. `(W/30, 1)` horizontal and `(1, H/30)` vertical, to keep only long lines [doc: OpenCV morph lines].
- Two ways to use it:
  1. **Subtract** long lines from the binary image so that boxes touching table rules separate from them. Keep the kernel length well above the checkbox side (e.g. ≥ 40 px) so the checkbox's own edges survive.
  2. Run the **same recipe with a short kernel** (≈ 0.7 × expected box side) to get a "short lines only" image. Checkbox outlines show up there as tiny closed rectangles, while text strokes mostly don't. This is essentially what `boxdetect` does internally [doc: boxdetect README].
- **Hough lines** (`HoughLinesP`) [doc: OpenCV Hough] are an alternative for grid detection. They are slower and need more parameter tuning than morphology for axis-aligned forms, so use them only if pages are skewed and you must deskew first.

### A3. Candidate generation
- `findContours(..., RETR_TREE or RETR_CCOMP, CHAIN_APPROX_SIMPLE)` [doc: OpenCV contours].
- Filters (all relative to page DPI; estimate scale from page width, e.g. a letter page at 2550 px ≈ 300 DPI):
  - bbox side within [~0.7, ~2.0] × expected side (≈ 10–25 px at the sample scale);
  - aspect ratio w/h in [0.8, 1.25];
  - `approxPolyDP(cnt, 0.02–0.05·perimeter)` yields 4 vertices, **or** squareness = contour area / bbox area > ~0.8 for the *outer* contour;
  - solidity / extent checks to reject filled bullets and letters like "O" or "D".
- **Hierarchy use** [doc: OpenCV hierarchy]: a hollow box gives an outer contour plus a child (the inner hole). Requiring a child, or a parent with an almost-square hole, rejects solid blobs. An "X" inside splits the hole into up to 4 triangle children, so **don't require exactly one child**.
- **Connected components** (`connectedComponentsWithStats`) is a faster alternative to contours for the size/aspect filter; use contours only for the final squareness test.
- **Template matching** (`matchTemplate` with `TM_CCOEFF_NORMED`) [doc: OpenCV template matching] with a synthetic empty-square template at 2–3 scales is a good *second* candidate source on these forms, because every box is the same size within a page. Its weaknesses: it is scale-sensitive, and it also fires on table cell corners. Use it to *recover* boxes that A2 lost to touching lines, not as the main detector.

### A4. Deduplication / NMS
- Outer and inner contours, multiple scales and template hits all produce near-duplicate boxes. Merge them with IoU NMS at ~0.3–0.5, preferring the candidate with the higher squareness score. `cv2.dnn.NMSBoxes` does this in one call.
- Also drop boxes nested inside a bigger accepted box of similar size (outer border vs. inner hole), and keep the outer one, since its bbox is the one expected in the output.

### A5. Fill classification
- Crop the box from the **binarized image before line removal**, then **erode/shrink inward** by ~15–25% of the side (or by the measured border thickness + 1–2 px) so the border itself doesn't count.
- `ink_ratio = count_nonzero(inner) / inner.size`. Empty boxes measure ≈ 0–0.03; an X or ✓ ≈ 0.10–0.40; solid fill ≈ 0.6+ [est; boxdetect's verbose example shows 0.0035 vs 0.42 [doc: boxdetect README]].
- Threshold: a fixed ~0.08–0.12, **or** run Otsu over the 1-D distribution of ratios for the page. Ratios are usually strongly bimodal per page, so this adapts to scan darkness automatically. Fall back to the fixed threshold when the page has only checked or only unchecked boxes (Otsu needs two modes).
- Extra cheap features if needed: diagonal pixel sums (X), count of ink components in the inner area, and the max run along each diagonal. Avoid over-engineering this without labels.

### A6. Typical failure modes (and mitigations)
| Failure | Why | Mitigation |
|---|---|---|
| Box shares an edge with a table rule | Contour merges with the grid, so the size filter rejects it | Long-line subtraction (A2.1); template-match recovery |
| Scribble or check mark extends outside the box | Outer contour is no longer square | Use the short-line image (A2.2) for detection, original binary for fill |
| Red watermark / blue shading | Thresholding turns overlays into ink, giving false fill or false boxes | Saturation mask before threshold (A1) |
| Letters "O", "D", "0", small table cells | Square-ish shapes | Size window, hollow-ness, require 4-vertex approx; cells are much larger |
| Thick X fills the hole, no child contour | Hierarchy test fails | Don't require a child; rely on outer squareness |
| Light, partial fill or faint check | Ink ratio near threshold | Per-page Otsu on ratios; report as checked when above the empty-box baseline + margin |
| Scale changes (different DPI) | Pixel-size filters are absolute | Normalize by page width or estimate the box side from the mode of candidate sizes |
| Skew > ~1° | Morphological line kernels miss tilted lines | Deskew with `minAreaRect` on text or Hough angle before everything else |

---

## B. Existing open-source checkbox libraries

| Project | What | Status |
|---|---|---|
| **boxdetect** (`karolzak/boxdetect`) | OpenCV pipeline for rectangular boxes on forms; `get_checkboxes()` returns `(x,y,w,h)`, a `contains_pixels` bool and a crop. Config: `width_range`, `height_range`, `scaling_factors`, `wh_ratio_range`, `group_size_range` (use `(1,1)` for checkboxes), `dilation_iterations` [doc] | **MIT**. Last PyPI release **1.0.2, 2023-01-18** (1.0.1 was 2022-12-27, 1.0.0 was 2020-07-16) [doc: PyPI JSON]. ~115 stars, 7 open issues, 48 commits [doc: GitHub]. Effectively unmaintained. Depends on opencv-python, numpy, imutils, pyyaml, scikit-learn |
| `jaswanth04/Checkbox_Detection` | OpenCV approach that needs a **blank template scan plus manually marked ROIs** | Demo-quality; template requirement rules it out for arbitrary pages [secondary] |
| `LynnHaDo/Checkbox-Detection` | YOLOv8 detection + classification models | Deep learning, out of scope here; worth noting as a "phase 2" path [secondary] |
| Tutorials (Fuzzy Labs, TDS "checkbox/table cell detection") | Same contour + morphology recipe as section A | Reference only [secondary] |

**Verdict:** read `boxdetect`'s pipeline for ideas (multi-scale, short-line morphology, grouping), but don't take it as a dependency. It is three years stale, adds scikit-learn for little benefit, and its fill test is the same one-liner you'd write anyway. Owning ~150 lines of OpenCV is more defensible in a take-home review.

Sketch with boxdetect (if used for a quick baseline):
1. `cfg = PipelinesConfig()`; set `width_range=height_range=(10,25)`, `wh_ratio_range=(0.8,1.25)`, `group_size_range=(1,1)`, `scaling_factors=[1.0]`.
2. `checkboxes = get_checkboxes(img, cfg, px_threshold=0.1)`.
3. Map `(x,y,w,h)` to `[x, y, x+w, y+h]` and `contains_pixels` to `is_checked`.

---

## C. Azure AI Document Intelligence: selection marks

- `prebuilt-layout` (v4.0, API `2024-11-30` GA) extracts selection marks per page with **`state` (`selected`/`unselected`), `polygon`, `confidence`** [doc: layout]. For images, the page `unit` is **`pixel`** [doc: layout], so the polygon maps directly to our bbox format (min/max of the 8 values).
- Input limits: 50×50 to 10,000×10,000 px [doc: layout].
- Pricing: about **US$10 per 1,000 pages** for Layout (from a third-party source, because the official page renders prices dynamically) [secondary: LiteLLM]. Free F0 tier: 500 pages/month [doc: pricing page].
- Privacy: inputs and results are stored temporarily and **deleted after 24 h**; a Delete Analyze Result API exists; processing stays in the resource's region [doc: privacy].
- Latency: asynchronous (submit, then poll for the result). Typically a few seconds per page [est].

Sketch:
1. `DocumentIntelligenceClient(endpoint, key).begin_analyze_document("prebuilt-layout", body=bytes)`.
2. For each `page.selection_marks`: `xs, ys = polygon[0::2], polygon[1::2]`, giving `bbox = [min xs, min ys, max xs, max ys]` and `is_checked = state == "selected"`.
3. Wrap it behind the same `detect(image) -> boxes` interface as A.

## D. AWS Textract: SELECTION_ELEMENT

- `AnalyzeDocument` / `GetDocumentAnalysis` returns `Block`s of type **`SELECTION_ELEMENT`** with `SelectionStatus` **`SELECTED` | `NOT_SELECTED`**, `Confidence`, and `Geometry.BoundingBox` / `Polygon` in **normalized 0–1 coordinates** [doc: Textract selectables].
- Boxes are attached to form KEY/VALUE pairs or to table CELLs. **With `FeatureTypes=["TABLES"]` only, boxes inside key-value pairs are omitted** [doc], so request both FORMS and TABLES to get full coverage.
- Pricing (US West Oregon, first 1M pages): Forms **$50/1k**, Tables **$15/1k**, plain text $1.50/1k. Free tier: 100 pages/month of Analyze for 3 months [doc: pricing].
- Privacy: Textract **may store and use inputs to improve AWS ML** unless you set an AWS Organizations AI-services opt-out policy [doc: FAQ].

Sketch: `analyze_document(Document={"Bytes": b}, FeatureTypes=["FORMS","TABLES"])`, then keep `BlockType == "SELECTION_ELEMENT"`, compute `bbox = [L*W, T*H, (L+w)*W, (T+h)*H]`, and set `is_checked = SelectionStatus == "SELECTED"`.

## E. Google Document AI: Form Parser

- The checkbox detector returns each checkbox as a **key-value pair**, keyed by the nearest text, with a `valueType` that marks it filled or unfilled. **Radio buttons aren't supported**, some checkboxes come back without a key, and parsing of unfilled values is described as weaker [doc: form parser].
- Geometry comes back as normalized bounding polys (multiply by page size).
- Pricing: **$30/1k pages** (first 1M/month), $20/1k after that [doc: pricing via search snippet].
- Setup is the heaviest of the three: GCP project, processor creation, and service-account auth.

Sketch: `process_document(name=processor, raw_document=...)`, then iterate `page.form_fields`, keep fields where `value_type` is `filled_checkbox`/`unfilled_checkbox`, and take the value's `bounding_poly.normalized_vertices` to get the bbox.

### Are managed services appropriate for this take-home?
- **As the core solution: no.** They need reviewer credentials and network access, send PII-heavy appraisal documents (addresses, borrower data) to a third party, cost money, and turn the exercise into "glue an SDK". The brief also asks for checkbox-level bboxes on *our* pixel grid. All three can provide that, but you inherit their misses with no way to tune.
- **As a discussed alternative: yes.** They are the obvious production option when volume is low and documents vary a lot. They are also a cheap way to **bootstrap pseudo-labels** for the 4 samples (and later a training set), which the classical pipeline can then be scored against. If one is integrated, it should be optional (env-var key), off by default.

---

## F. Vision-language models for checkbox bboxes

Evidence:
- **CheckboxQA** (Snowflake AI Research, arXiv 2504.10419, 2025) asks models to answer questions that depend on checkbox state. This is easier than localization, because no coordinates are needed. Scores (ANLS*): Qwen2.5-VL 72B 83.2, Qwen2.5-VL 7B 71.9, GPT-4o 66.7, Gemini 2.0 Pro 59.7, Gemini 2.0 Flash 54.4, GPT-4o mini 40.4; **human 97.5** [doc]. The authors attribute failures to checkboxes being small and visually subtle, needing layout context, and rarely appearing in training data. Observed errors include associating a box with the wrong label and answering from the text instead of the mark [doc].
- **Claude** docs: "coordinate and localization outputs are approximate"; counting "large numbers of small objects" may be inaccurate. Images above a long edge of 1568 px (2576 px on 4.7+ models) are **downscaled** [doc: Claude vision]. A 2550×3300 page scaled to 2576 px long edge loses ~22% of its resolution, so a 12 px box becomes ~9 px. On standard-tier models (1568 px) it becomes ~6 px.
- **Gemini** returns `box_2d` as `[ymin, xmin, ymax, xmax]` **normalized to 0–1000** [doc: Gemini image understanding]. One unit on a 3300 px page is 3.3 px, which is already ~25% of a checkbox side. The docs give no detection precision numbers.
- **Qwen2.5-VL** is trained to output **absolute pixel coordinates** with native dynamic resolution, and reports strong referring/grounding benchmarks [doc: Qwen2.5-VL report], but there is no small-object or document-checkbox localization benchmark. It is also a 7B–72B model, far outside a CPU latency budget.

Conclusion: no evidence that general VLMs produce reliable pixel-tight bboxes for ~50–100 tiny checkboxes per page. They are plausible as a **second-pass classifier on cropped, upscaled candidate boxes** ("is this box checked? yes/no"), but that adds cost, nondeterminism and an API key for a decision an ink-ratio threshold already gets mostly right. Mention this in the write-up; don't build it.

Sketch (if ever used as a verifier): run A, take boxes with `|ink_ratio − threshold| < margin`, upscale each crop 4–8× with some context, batch them in one request, ask for JSON `{"id": bool}`, and override only those boxes.

---

## Sources
- OpenCV, Image thresholding (Otsu, adaptive): https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- OpenCV, Extract horizontal and vertical lines by morphological operations: https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html
- OpenCV, Contours: getting started / hierarchy: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html , https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
- OpenCV, Template matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- OpenCV, Hough line transform: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
- boxdetect, GitHub README: https://github.com/karolzak/boxdetect
- boxdetect, PyPI metadata (versions, dates, license, deps): https://pypi.org/pypi/boxdetect/json
- Checkbox_Detection (template-based): https://github.com/jaswanth04/Checkbox_Detection
- Checkbox-Detection (YOLOv8): https://github.com/LynnHaDo/Checkbox-Detection
- Fuzzy Labs, checkbox detection blog: https://fuzzylabs.ai/blog/checkbox-detection/
- Azure Document Intelligence, layout model (selection marks, pixel units, v4.0 2024-11-30): https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout
- Azure Document Intelligence, data privacy (24 h retention): https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/document-intelligence/data-privacy-security
- Azure Document Intelligence pricing (F0 500 pages): https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/ ; Layout $10/1k (secondary): https://docs.litellm.ai/docs/providers/azure_document_intelligence
- AWS Textract, selection elements: https://docs.aws.amazon.com/textract/latest/dg/how-it-works-selectables.html
- AWS Textract pricing: https://aws.amazon.com/textract/pricing/
- AWS Textract FAQ (data use / opt-out): https://aws.amazon.com/textract/faqs/
- Google Document AI, Form Parser: https://docs.cloud.google.com/document-ai/docs/form-parser
- Google Document AI pricing: https://cloud.google.com/document-ai/pricing
- Turski et al., "Unchecked and Overlooked: Addressing the Checkbox Blind Spot in LLMs with CheckboxQA", arXiv 2504.10419: https://arxiv.org/html/2504.10419v1
- Anthropic, Claude vision (limits, resizing, limitations): https://platform.claude.com/docs/en/build-with-claude/vision
- Google, Gemini image understanding (box_2d 0–1000): https://ai.google.dev/gemini-api/docs/image-understanding
- Bai et al., Qwen2.5-VL Technical Report, arXiv 2502.13923: https://arxiv.org/abs/2502.13923
