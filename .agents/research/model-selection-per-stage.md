# Model selection per pipeline stage: locate, state, label

Researched 2026-10-10. Scope: pick the best permissively licensed model for each stage of the pipeline in
`.agents/spec.md` §6 (1 locate boxes, 2 classify checked/unchecked on crops, 3 OCR the label). Re-check the
models we chose (`nutrientdocs/form-field-v1-nano` + `form-field-v1-state`, branch `ticket-29-onnx`) against
newer releases. Measure them on more forms than the 4 samples, and verify the user's dataset list.
Constraints: CPU only, Docker on Railway (1 vCPU, 1-2 GB RAM), Python 3.13 + uv, onnxruntime at runtime (no
torch), and licenses that allow a public commercial API. AGPL or non-commercial models appear only as rejected or
eval-only.

This note builds on `learned-pipeline.md`, `deep-learning-and-data.md`, `public-checkbox-datasets.md` and
`classical-cv-and-services.md` and does not repeat them. Where a model was already rejected there, this note
links back and adds only new evidence.

Bracketed numbers like [3] point to the Sources section. **(probe)** means I measured it: onnxruntime 1.31.0,
`intra_op_num_threads=1`, Apple M4, scratch venv outside the repo, using the repo's own `YoloxLocator`,
`OpenCVLocator`, `StateCnnClassifier` and `FillRatioClassifier` classes at their default parameters unless
stated otherwise. **(est.)** means my own estimate. A Railway vCPU is probably 2-3x slower than one M4 core
**(est.)**. All P/R figures are box-level at IoU ≥ 0.5 with greedy one-to-one matching.

**About "el de Showlocks".** The transcription is unclear. Read phonetically, "Showlocks" is closest to
**Snowflake's**, i.e. CheckboxQA by Snowflake-Labs. In context, though, the user may have meant the Nutrient models
we picked. This note covers both. CheckboxQA is a non-commercial QA benchmark with no boxes, so it cannot be a
stage model at all (§4). The Nutrient models are re-tested below against every permissive alternative I could
run.

---

## TL;DR

- **Locate: keep `form-field-v1-nano`, but run it at a coarser tile (`tile_frac` 1.0 instead of 0.6).** It is
  still the only permissive detector that fits the budget and finds boxes on our scans. **(probe)** On the 4
  samples, tile 1.0 gives P 0.966 / R 0.986 at 67 ms/page, against P 0.953 / R 0.993 at 182 ms for the current
  0.6. On 354 CommonForms-derived pages (vlm-v2 benchmark), R rises from **0.587 to 0.749**. On 80 mini-CommonForms
  pages it rises from 0.581 to 0.689. Nano's remaining weak spots are radio circles, colored boxes (our
  overlay removal erases them) and very small boxes. It reaches only about 0.7 recall on generic forms.
  **Follow-up (2026-10-10, ticket #29): kept at 0.6 for now.** Run end to end with the `cnn` classifier on the
  samples, tile 1.0 boxes are looser (mean IoU vs ground truth 0.77-0.87, p90 corner error 3-7 px; tile 0.6:
  0.83-0.90, 2-5 px), and the worse crops raise state errors from 1 to 5 of ~286 boxes (3 with `pad_frac` 0.1).
  Tile 0.8 gives no gain over either. The switch is left to the default-pipeline checkpoint (#22), measured end
  to end with `evaluate.py` **(probe)**.
- **The only new permissive detector that runs on onnxruntime, `Frooodle/ffdetr-int8`** (int8 ONNX of
  `jbarrow/FFDetr`, RF-DETR-M, Apache-2.0 lineage) [3][4], **fails on our scans**. **(probe)** It reaches R 0.44 on
  the samples (0.00 on sample-003) and **1.47 GB peak RSS** for one page, at about 1.1 s/page. It wins on clean
  born-digital forms (mini-CommonForms R 0.95, though CommonForms train leakage is possible). That makes it an
  offline teacher, not a runtime model. Its filled-form fine-tune `pdf-net/ffdetr-filled` (Apache-2.0) [5] ships
  only a pickled `.pth`, so I did not run it.
- **Docling egret-medium now has a community ONNX** (`ningpp/...-ONNX`, Apache-2.0) [12]. **(probe)** It has the
  same failure as heron in `learned-pipeline.md` §2: boxes wrap the checkbox plus its label word, giving P 0.10 /
  R 0.04 on the samples at IoU 0.5. Rejected. PP-DocLayoutV3 has no checkbox class [14]. Rejected.
- **State: keep `form-field-v1-state`.** **(probe)** It scores 0.997 on the 288 sample boxes and **0.982 on 2,728
  checkbox and radio crops** from the vlm-v2 benchmark (empty, printed-filled and handwritten), at 0.3-0.5
  ms/crop. The only other permissive state model, `wendys-llc/checkbox-classifier`, trained on UI checkboxes,
  reaches checked recall **0.08-0.11** on forms. The `fill_ratio` baseline is perfect on the samples but drops
  to 0.709 on generic forms. Improve the CNN by fine-tuning it (its `state_cnn.pt` is published) rather than
  swapping it.
- **Label: keep RapidOCR recognition-only with the default PP-OCRv6 *small* model.** **(probe)** On the 251
  labelled sample boxes it gets 0.880 exact/prefix match at 10.6 ms/crop, the best accuracy for the cost:
  v6-medium also scores 0.880 but takes 6x longer, while v6-tiny (0.749) and PP-OCRv5-en-mobile (0.777) are worse.
  On generic forms the weak part is *where* the label is, not the OCR (§3).
- **Datasets:** the user's list is mostly right. Three corrections: CommonForms names the class `CheckBox`
  (id 1), not "Choice Button"; AutoFormBench images in the repo are **200 DPI**, not 300, and its checkboxes are
  points; Roboflow downloads need an account/API key (§4). Two sets matter more than anything on the list:
  **mini-CommonForms** (Apache-2.0, curated boxes) for locate, and **nutrientdocs/form-field-vlm-v2-benchmark**
  (CC BY 4.0, bbox + checked state + label) for state and label. Both were probed here.

---

## 1. Stage 1: locate

### 1.1 Candidates, ranked

| # | Model | License (primary source) | Input / scale handling | Size | CPU latency (probe, 1 thr M4) | Reported metric | Fit | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **form-field-v1-nano** (YOLOX-Nano) [1] | Apache-2.0 card; YOLOX Apache-2.0 [29]; ONNX has no license metadata **(probe)** | 640² letterbox, decode baked in; we tile | 3.7 MB (fp16 1.9 MB) | 67 ms (tile 1.0) / 182 ms (tile 0.6) on 2550×4200; 28-63 ms on 1.1-1.7k px pages | Card mAP50-95 empty 0.378 / filled 0.550 / handwritten 0.514; Choice 0.482 on empty [1] | Excellent | **Keep. Switch the default to tile 1.0.** |
| 2 | OpenCV locator (repo baseline) | ours | Rule-based, tuned on the 1004 samples | — | 98 ms samples; 30-46 ms smaller pages | — | Excellent | Keep as no-weights fallback. Strong on the 1004s, weak on generic forms (§1.2). |
| 3 | `Frooodle/ffdetr-int8` (FFDetr, RF-DETR-M) [4] | Apache-2.0 card; lineage FFDetr / RF-DETR / DINOv2 / CommonForms, all Apache-2.0 [3][4]; ONNX metadata only quantizer tags **(probe)** | Fixed 1×3×1024×1024, square squash, ImageNet norm, outputs `dets [300,4]` cxcywh 0-1 + `labels [300,4]` logits; class 1 = ChoiceButton **(probe)** | 37 MB | 1.1-1.4 s full page; 6.3 s tiled | Base FFDetr on 300 synthetic CommonForms val pages: P 0.84 / R 0.53, filled-scan R 0.02 [5] | Poor: **1.47 GB peak RSS for one page (probe)** | Reject for runtime; offline teacher for born-digital pages |
| 4 | `pdf-net/ffdetr-filled` [5] | Apache-2.0 card | RF-DETR-M at 1024 | 135 MB `.pth` only | not run (pickle) | Card: overall P 0.87 / R 0.77, filled-scan R 0.86 (vs 0.02 base), AP50 0.815 [5] | Needs torch + rfdetr, or our own ONNX export | Best permissive candidate for filled scans on paper. Offline teacher; card itself says it is an "interim checkpoint" [5] |
| 5 | Docling egret-medium ONNX (`ningpp`) [12][13] | Apache-2.0 | Dynamic H/W in the ONNX **(probe)**; preprocessor squashes to 640² [12] | 78 MB | 183 ms full page; 1.1 s tiled | No per-class checkbox AP published (see `learned-pipeline.md` §2) | Poor | **Reject.** Box wraps the label: samples P 0.10 / R 0.04 (IoU 0.5), R 0.18 at IoU 0.3 **(probe)** |
| — | Docling heron (and fp16/int8 community ONNX) | Apache-2.0 | 640² squash | 171 MB | — | — | — | Already rejected in `learned-pipeline.md` §2; egret confirms it |
| — | PP-DocLayoutV3 / PP-DocLayout_plus-L [14] | Apache-2.0 | — | — | — | — | — | **No checkbox class** (25 labels: text, table, seal, …) [14]. Reject |
| — | FFDNet-S/L ONNX [26] | **AGPL-3.0 in ONNX metadata** | 1216² | 38/102 MB | — | — | — | Already rejected (`learned-pipeline.md` §3) |
| — | wendys-llc/checkbox-detector, takumi123xxx/pdfme, LynnHaDo/linhdo, AutoFormBench YOLOv11/26, `suv11235/*`, `bugsiesegal/form-field-detection-yolo` | Ultralytics-trained → AGPL [24]; several have no license; AutoFormBench data is non-commercial [18] | — | — | — | AutoFormBench YOLOv11 checkbox P 0.855 / R 0.800 at 20% tolerance [18] | — | Reject (AGPL, non-commercial, or no provenance, and mostly `.pt` pickles) |
| — | `nutrientdocs/form-field-v1-small/-medium`, `form-field-vlm-v2` [11] | `nutrient-commercial`, weights not downloadable | — | — | — | VLM-v2 hybrid end-to-end F1 0.565 [11] | — | Unusable |

The HF hub search (2026-10-10) for `checkbox`, `form-field`, `commonforms`, `ffd`, `jbarrow`, `docling-layout`,
`PP-DocLayout` and `omr` found no other permissive checkbox or form-field detector. The only new ones since the
existing notes are #3, #4 and the egret ONNX ports [3][4][5][12].

### 1.2 Probe results

Datasets (details in §4). **samples**: our 4 scans with hand-made ground truth, 288 boxes. **mini-CF**: the 80
val/test pages in the mini-CommonForms sample parquet, `choice` class, 1081×1400 px. **The sample parquet repeats
175 of 571 boxes (IoU > 0.7), so I deduplicated it to 396.** **CF-test**: 48 pages fetched from the raw
CommonForms test split via the datasets-server; only 8 have boxes (130). **vlm-v2**: all 354 pages,
`choice_checkbox` + `choice_radio`, 2,663 boxes after deduplication, 1654-1700 px wide. Score thresholds: nano
0.2 (repo default), FFDetr 0.3, egret 0.3.

| Locator **(probe)** | samples P / R | ms/page | mini-CF P / R | ms | vlm-v2 P / R | ms | CF-test P / R |
|---|---|---|---|---|---|---|---|
| nano, tile 0.6 (current default) | 0.953 / **0.993** | 182 | 0.780 / 0.581 | 145 | 0.732 / 0.587 | 165 | 0.449 / 0.238 |
| **nano, tile 1.0** | **0.966 / 0.986** | **67** | 0.756 / 0.689 | 52 | **0.795 / 0.749** | 63 | 0.660 / 0.254 |
| nano, tile 1.5 (≈ one pass) | 0.962 / 0.712 | 67 | 0.760 / 0.694 | 28 | 0.794 / 0.724 | 35 | — |
| nano, tile 0.4 | — | — | 0.862 / 0.252 | 358 | — | — | — |
| nano, tile 0.6 + 1.0 merged (NMS 0.3) | 0.932 / 1.000 | 250 | 0.724 / 0.697 | 198 | 0.739 / 0.770 | 228 | 0.437 / 0.238 |
| nano, tile 1.0, no colour-overlay removal | — | — | 0.777 / 0.684 | 58 | — | — | — |
| OpenCV (repo) | **0.989** / 0.969 | 98 | 0.695 / 0.270 | 30 | 0.469 / 0.367 | 35 | 0.040 / 0.208 |
| FFDetr int8, full page | 0.977 / 0.441 | 1085 | 0.828 / **0.949** | 1093 | 0.752 / 0.676 | 1203 | **0.766 / 0.454** |
| FFDetr int8, tiled 0.6 | 0.948 / 0.507 | 6272 | — | — | — | — | — |
| egret-m ONNX, full / tiled 0.6 | 0.096 / 0.042 · 0.157 / 0.177 | 183 · 1096 | — | — | — | — | — |

vlm-v2 by render condition, nano tile 1.0: empty P 0.701 / R 0.639, filled 0.852 / 0.732, handwritten
0.848 / 0.844. With tile 0.6: 0.623 / 0.496, 0.823 / 0.616, 0.775 / 0.621 **(probe)**.

What the numbers mean:

- **Tile 0.6 was tuned on 4 pages and overfits them.** At 1.0 the samples lose 2 boxes (284 vs 286) and gain
  precision. Generic forms gain 10-16 recall points. The run is also 2.7x faster, because a 2550×4200 page is cut into 2
  windows instead of 8. Merging both scales buys only +2 points on vlm-v2 for about 3.5x the cost.
- **Nano's misses on mini-CF**, inspected by eye on a crop sheet, are mostly **radio circles**, very small
  `☐` glyph boxes, a few **red-outlined boxes** (our `to_clean_gray` turns saturated outlines white), and GT
  boxes that have no visible square. Turning overlay removal off does not change the total (0.684 vs 0.689).
  Recall falls at tile 0.4: nano wants boxes of about 10-25 px in its 640 input, and upscaling CommonForms pages
  pushes them past that.
- **The raw CommonForms test GT is not a fair recall target.** Labels come from PDF widgets, and many widgets
  have no printed square. On `cf_28860` (ACRM registration form) the GT has 48 choice boxes, but only about 6
  squares are drawn **(probe, by eye)**. Every model scores R < 0.46 there. This is the label noise
  mini-CommonForms was built to fix [9].
- **FFDetr leakage caveat.** FFDetr was trained on CommonForms train [3]. The mini-CF sample's `val`/`test`
  labels are mini-CF's own splits, and I could not confirm they map to CommonForms' held-out pages. Its 0.949 on
  mini-CF may be optimistic. Its CF-test score (0.454 on true held-out pages) and its scan score (0.441) are the
  safer reads.
- **The OpenCV locator is a 1004 specialist.** It is the most precise on the samples but finds a quarter to a
  third of the boxes on other forms. That is expected for thresholds tuned on 4 pages.

### 1.3 Verdict, stage 1

1. **Now:** tile 1.0 finds more boxes (evidence above), but its looser bboxes cost state accuracy end to end
   (see the TL;DR follow-up). `tile_frac` stays 0.6 until #22 compares both end to end with `evaluate.py`.
2. **Next, if generic-form recall matters:** fine-tune nano. YOLOX is Apache-2.0 [29], and Nutrient publishes
   `best_ckpt.pth` + `exp.py` [1]. Train on **mini-CommonForms train** (Apache-2.0, 179,500 choice boxes [9]),
   rendered official 1004/1004MC/1004C pages (`public-checkbox-datasets.md` §1), and colored or radio examples.
   Use FFDetr or `ffdetr-filled` offline as a pseudo-labeler for scans [5]. Export to ONNX as today. The
   `.pth` is a pickle from a known publisher. Load it only offline in a dev env, with `weights_only=True`.
3. Do not adopt FFDetr, egret/heron, FFDNet or any Ultralytics model at runtime.

## 2. Stage 2: checked state

### 2.1 Candidates, ranked

| # | Model | License | Input | Size | CPU latency (probe) | Reported metric | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **form-field-v1-state** [2] | Apache-2.0 | 40×160 gray letterbox, batched ONNX `crop [n,1,40,160]` → `logits [n,2]` **(probe)** | 0.38 MB | 0.3-0.5 ms/crop | Card acc 0.981 vs ink heuristic 0.915 (synthetic split) [2] | **Keep**; fine-tune `state_cnn.pt` for the stray-stroke case |
| 2 | `fill_ratio` (repo baseline) | ours | ink ratio inside the box | — | 0.6-0.7 ms/crop | — | Keep as baseline/fallback only |
| 3 | `wendys-llc/checkbox-classifier` (EfficientNet-B0) [15] | Apache-2.0 | 224² RGB; safetensors (needs torch/transformers or our own ONNX export) | 16 MB | 58-68 ms/crop in torch | Card val acc 0.97 on UI checkboxes (`wendys-llc/chkbx`) [15] | **Reject**: UI domain, misses almost every checked form box |
| — | `hermitkk/omr-checkbox-classifier` [16] | **AGPL-3.0** | ONNX, exam sheets | — | — | — | Reject (license) |
| — | VLM checkbox fine-tunes (`meghnagera15/paligemma-…`, `PriyanshuMaheshwari/checkbox-vlm-qwen`, `aryan2723/qwen2vl-checkbox-lora`, `SnehaElla/checkbox-clip-finetuned`, `Apara17/checkbox-vit-model`) | no license tag on any | — | — | — | — | Reject (no license; VLMs are off-budget, see `deep-learning-and-data.md` §6) |
| — | Docling egret/heron `Checkbox-Selected/Unselected` | Apache-2.0 | whole page | — | — | — | Reject; state comes with the bad boxes of §1 |

### 2.2 Probe results (crops cut from ground-truth boxes, so this isolates stage 2)

vlm-v2 truth = `value == "checked"`. The `state` field is stale on handwritten pages: 66 checkboxes and 77 radios
have `state=unchecked, value=checked`. I checked a crop sheet and they are visibly ticked **(probe)**.

| Classifier **(probe)** | samples (288) acc | vlm-v2 (2,728) acc | vlm-v2 errors (FP / FN) | checked recall / precision | ms/crop |
|---|---|---|---|---|---|
| **cnn** (repo default: clean gray, threshold 0.5 + ambiguous margin 0.2) | **0.997** (1 FP) | **0.982** | 40 / 10 | 0.982 / 0.931 | 0.40-0.49 |
| cnn, no ambiguous margin (threshold 0.5) | 0.997 | 0.979 | 52 / 5 | 0.991 / 0.913 | 0.36-0.48 |
| cnn, plain gray (no overlay removal) | 0.997 | 0.976 | 55 / 10 | 0.982 / 0.907 | 0.26-0.27 |
| fill_ratio | **1.000** | 0.709 | 791 / 2 | 0.996 / 0.409 | 0.57-0.71 |
| wendys EfficientNet-B0 | 0.712 | 0.799 | 41 / 507 | **0.077** / 0.506 | 58-68 |

cnn on vlm-v2 by condition: empty checkbox 0.991, empty radio 0.960, filled checkbox 0.988, filled radio 0.966,
handwritten checkbox 0.975, handwritten radio 0.985.

- I looked at a sheet of the first 48 cnn disagreements on vlm-v2. **Roughly a third are real model errors:**
  ticks scoring 0.5-0.7 that the ambiguous band sends to unchecked, a handwritten stroke or a printed digit
  inside an empty box (the same failure as the known stray-stroke case on our samples), and faint or partial
  ticks. **The rest are label noise or genuinely ambiguous glyphs:** solid `■` squares (11 of 48), pre-printed
  selected radios marked unchecked, `☒`/`✔` boxes marked unchecked, and GT boxes on pictures or text
  **(probe, by eye)**. The 0.982 is a lower bound on rendered pages.
- **Leakage caveat:** vlm-v2 is drawn from Nutrient's `commonforms-synth-v2` [10], and `form-field-v1-state` is
  also trained on Nutrient's synthetic renders [2]. Its marks may look like the state model's training marks,
  so do not read 0.982 as scan accuracy. Real-scan state accuracy is only measured on our 4 samples (0.997).
- `fill_ratio` collapses on generic forms because the CommonForms GT box is the widget rect, not the drawn
  square. Borders and nearby text then count as ink (791 false positives). That is a box-convention mismatch as
  much as a weakness of the method. On the 1004s it is perfect.

### 2.3 Verdict, stage 2

Keep `cnn`. If more accuracy is needed, **fine-tune `state_cnn.pt`** [2] offline (torch in a dev group only),
then re-export to ONNX. Add three things to the training data: hard negatives (strokes, digits and lines
crossing empty boxes), solid-`■` and pre-printed glyphs, and real scans. Real-scan crops can come from NAF
(CDLA-Permissive) and the public-domain or CC BY Roboflow sets in `public-checkbox-datasets.md`. Keep vlm-v2 and
our samples as **eval only**. vlm-v2 is CC BY 4.0, so training on it is legal with attribution [10], but it is
the only clean public state benchmark.

## 3. Stage 3: label OCR

### 3.1 Candidates

| # | Option | License | Runtime | Probe / reported | Verdict |
|---|---|---|---|---|---|
| 1 | **RapidOCR 3.10, rec-only, PP-OCRv6 *small* (default)** [22] | Apache-2.0 (code); PaddleOCR models Apache-2.0 | onnxruntime, pip only | **(probe)** samples: 0.880 exact/prefix, mean similarity 0.922, 10.6 ms/crop | **Keep** |
| 2 | RapidOCR, PP-OCRv6 medium | same | same | **(probe)** 0.880 / 0.909, 62.8 ms/crop | Not worth 6x the cost |
| 3 | RapidOCR, PP-OCRv5 `en` mobile | same | same | **(probe)** 0.777 / 0.852, 19.7 ms/crop | Worse |
| 4 | RapidOCR, PP-OCRv6 tiny | same | same | **(probe)** 0.749 / 0.909, 23.1 ms/crop (not faster on M4) | Worse |
| 5 | OnnxTR 0.9.0 [23] | Apache-2.0 | onnxruntime | not probed; see `learned-pipeline.md` §1 | Fallback only |
| — | PaddleOCR 3.7, docTR, Tesseract | — | — | — | Already rejected in `learned-pipeline.md` §1 (numpy pin, torch, tessdata/apt) |
| — | pypdfium2 text layer | Apache/BSD | already a dependency | `learned-pipeline.md` §6 | Use first for born-digital PDFs |

Setup **(probe)**: a strip to the right of each GT box, 1.5 × side tall and up to 12 × side wide, cut at the
next box on the same row. Run `eng(crop, use_det=False, use_cls=False)` and compare lower-cased alphanumerics
with the GT `label`. "Exact/prefix" counts OCR text equal to the label or starting with it.

- **Generic forms are a placement problem, not an OCR problem.** On 608 English checkbox labels from vlm-v2
  empty pages, the same setup scores only 0.260 exact/prefix (mean similarity 0.416; v5-en is 0.169). The gold
  `label` there is written as a field name (for example "I, (Parent/Legal Guardian Name);"), often sits left of
  or above the box, and is sometimes paraphrased [10]. So this number measures the right-of-box assumption and
  the label definition more than recognition **(probe)**. Before adding OCR models, add placement rules (look
  right, then left, then above; stop at rules/boxes).

### 3.2 Verdict, stage 3

Keep RapidOCR rec-only on PP-OCRv6 small, which is the default config. Spend effort on strip placement. Use the
pypdfium2 text layer when the input is a born-digital PDF.

## 4. Datasets: verified facts

Verified 2026-10-10 against each primary page. The "User said" column quotes the user's list.

| Dataset | User said | Verified (primary source) | Corrections |
|---|---|---|---|
| **CommonForms** [7][8] | form-field bboxes (Text Input, Choice Button, Signature), no state, ~487k pages, Apache-2.0 | Apache-2.0 (card YAML). train 435,698 / valid 18,195 / test 33,061 = **486,954 pages** (~55k docs). Classes in the data: **`0 Text, 1 CheckBox, 2 Signature`**. COCO `[x,y,w,h]` px. No state. 153 GB, test = 24 parquet shards of about 470 MB. A few rows can be fetched through the datasets-server `rows` API [8]. | Class is named `CheckBox` in the data. **Labels are PDF widget rects, often with no printed square** **(probe)**, so it is noisy as a recall target. |
| **mini-CommonForms** (not on user's list) [9] | — | Apache-2.0. 47,355 pages (train 27,338 / val 7,400 / test 12,617). Classes `1 text, 2 choice, 3 sig`. 365,253 choice boxes. Full set is 3.9 GB test tar + JSON. A 200-page sample parquet (34 MB) exists. | **The sample parquet duplicates ~31% of choice boxes** **(probe)**. Deduplicate before scoring. |
| Roboflow `myproject-gemsb/checkboxes-2-dyp1n` [17] | bbox + state, 6 inconsistent classes, 1.8k imgs, CC BY 4.0 | 1,803 images, CC BY 4.0, 1 version, 0 models. Classes `check`, `Checkbox-Checked`, `Checkbox-UnChecked`, `checked`, `Checked`, `unchecked`. | Correct. `check` is undefined. |
| Roboflow `tempqqaq/checkbox_v2` [17] | 198 imgs, CC BY 4.0 | 198, CC BY 4.0, `checked`/`unchecked`, 1 version | Correct |
| Roboflow `ujiboo/checkbox-detector-zxwoh` [17] | 133 imgs, CC BY 4.0 | 133 on the project page (a version with 298 also exists), CC BY 4.0, `checked`/`unchecked`. 2 "Roboflow Instant" hosted models with undisclosed architecture and no weights. | Count depends on version |
| Roboflow `office-8ecbg/checkbox-detection-jiid0` [17] | 72 imgs, CC BY 4.0 | 72, CC BY 4.0, `checked`/`unchecked`, **0 dataset versions (no export possible)**. 1 hosted Instant model. | Not downloadable |
| (all Roboflow) | — | Pages sit behind Cloudflare. Export needs a Roboflow account + API key [28]. | **Skipped in probes (API key)** |
| **AutoFormBench** [18] | 407 real forms at 300 DPI, checkboxes as points, non-commercial, YOLOv11 vs OpenCV | 407 forms (506 collected, 99 dropped). Classes `checkbox`, `line`, `box`. Checkboxes are **points** (YOLO export uses fixed 0.02×0.02 boxes). Repo image `form100.png` is 1654×2339 = **A4 at 200 DPI**. Site license: non-commercial research/education only. Repo has no LICENSE. arXiv CC0 covers the paper only. Results at 20% tolerance: YOLOv11 P 0.855 / R 0.800, YOLOv8 0.820 / 0.826, OpenCV 0.822 / 0.687, YOLO26-s 0.944 / 0.622. Weights are Ultralytics `.pt` only. Repo about 625 MB. | DPI is 200. Eval-only at best. Boxes are not IoU-scorable. |
| **CheckboxQA** [19] | 88 docs, 579 QA, CC BY-NC | 88 docs, 579 QA (`data/gold.jsonl`). LICENSE **CC BY-NC 4.0**. "Intended solely for evaluation". No boxes. PDFs fetched from DocumentCloud by a script. | Correct. Not usable for any stage except offline end-to-end QA eval |
| **LynnHaDo/Checkbox-Detection** [20] | YOLOv8 on RVL-CDIP, AGPL, data not public | AGPL-3.0. Ultralytics YOLOv8s, classes `unchecked/checked/block`. Synthetic pastes on RVL-CDIP (10k+), 150 human-annotated val docs. Data "not publicly available". `.pt` weights in HF Space `linhdo/checkbox-detector`. No numeric metrics. | Correct |
| **nutrientdocs/form-field-vlm-v2-benchmark** (not on user's list) [10] | — | CC BY 4.0. **354 pages** (113 empty / 116 filled / 125 handwritten), 8,220 fields, of which 2,026 `choice_checkbox` + 705 `choice_radio`, with `box`, `state`, `value`, `label`. 142 MB parquet. | Use `value` as truth: `state` is stale on 143 handwritten fields **(probe)** |
| nutrientdocs/form-field-v1-benchmark [30] | — | CC BY 4.0. 5,914 pages, Text/ChoiceButton/Signature. **2.68 GB single parquet** | Not downloaded (size) |

Which set evaluates which stage:

| Stage | Train (permissive) | Eval |
|---|---|---|
| Locate | mini-CommonForms train (Apache-2.0); rendered official 1004 forms; Roboflow CC BY / PD sets (API key) | our samples; mini-CF val/test; vlm-v2 (incl. handwritten); AutoFormBench (point match, non-commercial, eval-only) |
| State | vlm-v2 (CC BY 4.0, but better kept for eval); Roboflow checked/unchecked sets; NAF; synthetic marks | our samples; vlm-v2 (2,728 crops with state) |
| Label | none needed (pretrained OCR) | our samples (`label` in GT JSON); vlm-v2 `label` (semantic, placement-sensitive) |
| End-to-end | — | CheckboxQA (CC BY-NC, eval only) |

## 5. Recommendation per stage

| Stage | Decision | Evidence | Next step |
|---|---|---|---|
| Locate | **Keep nano; tile 1.0 vs 0.6 decided in #22** | Samples R 0.986 / P 0.966 at 67 ms vs R 0.993 / P 0.953 at 182 ms; vlm-v2 R 0.749 vs 0.587; mini-CF R 0.689 vs 0.581; but end to end with `cnn` 5 vs 1 state errors on the samples **(probe)** | Fine-tune nano (YOLOX, Apache) on mini-CF train + rendered 1004s, with FFDetr / ffdetr-filled as offline teacher |
| State | **Keep form-field-v1-state** | 0.997 samples, 0.982 on 2,728 generic crops; alternatives 0.709-0.799 **(probe)** | Fine-tune `state_cnn.pt` with stray-stroke / digit / `■` negatives and real scan crops |
| Label | **Keep RapidOCR rec-only, PP-OCRv6 small** | 0.880 exact/prefix at 10.6 ms/crop; no variant better per ms **(probe)** | Improve strip placement (right → left → above) |

## 6. Unknowns

- **Real-scan accuracy beyond our 4 pages.** No public, permissively licensed set of *scanned* forms with box +
  state could be downloaded without an API key. The Roboflow sets need an account. NAF is 790 MB and was not
  probed.
- **`ffdetr-filled` on our scans**, and fp32 vs int8 FFDetr. The only weights are pickled `.pth`, and I did not
  run unknown pickles. Exporting it to ONNX in a sandbox would answer this. Its card reports filled-scan R 0.86
  on synthetic data [5].
- **Leakage** between Nutrient's training renders and vlm-v2, and between FFDetr's training set and the
  mini-CF sample pages. Neither card lists page IDs.
- **Whether tile 1.0 holds for 1004 scans at other DPIs.** The probe sets cover 1.1k-2.6k px widths. A 150-DPI
  1004 (1275 px wide) behaves like the mini-CF pages, but that is **(est.)**, not measured.
- **Railway timings.** All latencies are one M4 core. Expect 2-3x slower **(est.)**.

---

## Sources

1. form-field-v1-nano (Apache-2.0; YOLOX-Nano; card metrics, `best_ckpt.pth`, `exp.py`): https://huggingface.co/nutrientdocs/form-field-v1-nano
2. form-field-v1-state (Apache-2.0; 96k-param CNN; `state_cnn.pt`): https://huggingface.co/nutrientdocs/form-field-v1-state
3. FFDetr (Apache-2.0; RF-DETR on CommonForms): https://huggingface.co/jbarrow/FFDetr
4. ffdetr-int8 ONNX (Apache-2.0; lineage table): https://huggingface.co/Frooodle/ffdetr-int8
5. ffdetr-filled (Apache-2.0; filled-form fine-tune, `.pth` only, interim): https://huggingface.co/pdf-net/ffdetr-filled
6. commonforms FFDetrDetector (RFDETRMedium, 1024, `id_to_cls`): https://github.com/jbarrow/commonforms/blob/main/commonforms/inference.py
7. CommonForms dataset card + API: https://huggingface.co/datasets/jbarrow/CommonForms ; paper https://arxiv.org/abs/2509.16506
8. HF datasets-server rows API (CommonForms test): https://datasets-server.huggingface.co/rows?dataset=jbarrow/CommonForms&config=default&split=test&offset=0&length=1
9. mini-CommonForms: https://huggingface.co/datasets/omoured/minicommonform ; https://github.com/moured/mini-commonforms ; https://arxiv.org/abs/2609.23679
10. form-field-vlm-v2-benchmark (CC BY 4.0, `benchmark_meta.json`): https://huggingface.co/datasets/nutrientdocs/form-field-vlm-v2-benchmark
11. form-field-vlm-v2 (nutrient-commercial): https://huggingface.co/nutrientdocs/form-field-vlm-v2 ; form-field-v1-small: https://huggingface.co/nutrientdocs/form-field-v1-small
12. Docling egret-medium ONNX port (Apache-2.0, config/preprocessor): https://huggingface.co/ningpp/docling-layout-egret-medium-ONNX
13. Docling egret-medium (Apache-2.0): https://huggingface.co/docling-project/docling-layout-egret-medium
14. PP-DocLayoutV3 labels (`config.json` `id2label`): https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/blob/main/config.json ; PP-DocLayout_plus-L: https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_onnx
15. wendys-llc/checkbox-classifier (Apache-2.0, EfficientNet-B0, UI data): https://huggingface.co/wendys-llc/checkbox-classifier
16. hermitkk/omr-checkbox-classifier (AGPL-3.0): https://huggingface.co/hermitkk/omr-checkbox-classifier
17. Roboflow: https://universe.roboflow.com/myproject-gemsb/checkboxes-2-dyp1n · https://universe.roboflow.com/tempqqaq/checkbox_v2 · https://universe.roboflow.com/ujiboo/checkbox-detector-zxwoh · https://universe.roboflow.com/office-8ecbg/checkbox-detection-jiid0
18. AutoFormBench: https://tempgaurab.github.io/Detect_Regions_in_PDF/ · https://github.com/TempGaurab/Detect_Regions_in_PDF · https://arxiv.org/abs/2603.29832
19. CheckboxQA (CC BY-NC 4.0): https://github.com/Snowflake-Labs/CheckboxQA
20. LynnHaDo/Checkbox-Detection (AGPL-3.0): https://github.com/LynnHaDo/Checkbox-Detection ; Space: https://huggingface.co/spaces/linhdo/checkbox-detector
21. Hub searches used (2026-10-10): https://huggingface.co/api/models?search=checkbox · https://huggingface.co/api/models?search=form-field · https://huggingface.co/api/models?search=docling-layout · https://huggingface.co/api/models?search=PP-DocLayout
22. RapidOCR 3.10.0 (Apache-2.0; PP-OCRv6 tiny/small/medium, PP-OCRv5 en mobile in `default_models.yaml`): https://pypi.org/project/rapidocr/ ; https://github.com/RapidAI/RapidOCR
23. OnnxTR 0.9.0 (Apache-2.0): https://pypi.org/project/onnxtr/
24. Ultralytics licensing (trained models AGPL): https://www.ultralytics.com/license
25. Other HF checkbox repos checked: https://huggingface.co/suv11235/checkbox-detection-yolov8-resnet · https://huggingface.co/bugsiesegal/form-field-detection-yolo · https://huggingface.co/takumi123xxx/pdfme-form-field-detector · https://huggingface.co/wendys-llc/checkbox-detector
26. FFDNet-S ONNX: https://huggingface.co/jbarrow/FFDNet-S-cpu
27. RF-DETR (Apache-2.0 N-L): https://github.com/roboflow/rf-detr
28. Roboflow Universe download docs: https://docs.roboflow.com/universe/download-a-universe-dataset
29. YOLOX (Apache-2.0): https://github.com/Megvii-BaseDetection/YOLOX
30. form-field-v1-benchmark (CC BY 4.0, 2.68 GB parquet): https://huggingface.co/datasets/nutrientdocs/form-field-v1-benchmark
