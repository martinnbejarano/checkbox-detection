# Write-up

## Summary

The service finds the checkboxes on a page, says whether each one is checked and, on request, reads its label. The default pipeline straightens the page, locates the boxes with a fine-tuned YOLOX-Nano, classifies each one with a fine-tuned 96k-parameter CNN and reads labels with RapidOCR. It all runs on CPU through onnxruntime, with no external service.

On the 27 sample pages (1,370 boxes) it scores 99.3% end-to-end. On 15 third-party pages of other forms (419 boxes) it scores 71.8% and calls no empty box checked. Neither set was used for tuning. A page takes about 1.4 s on one thread, and 1.2 s of that goes to reading its orientation. The biggest gap is data, since no real appraisal scans were available for training.

## How I got here

I kept each step below only if it moved a measured number.

### 1. Measure first

Before developing any detector, I labeled the 4 original samples (288 bounding boxes) by hand, under a written policy. A box is checked when an intentional mark sits mostly inside it; a stray stroke, a watermark or a colored cell fill does not count. I also put aside an external set of 15 pages labeled by third parties (419 boxes) and never tuned on it.

`make eval` runs every locator × classifier combination on both sets. The headline metric counts a box only if it was found at IoU ≥ 0.5 and its state is right.

### 2. A classical baseline

OpenCV + `fill_ratio` came first because it needs no data or weights and every threshold can be read.

On the 4 samples it scored 96.9% end-to-end, 279 of 288 boxes. All 9 misses were boxes it did not find, since every box it found had the right state. Five of them touch a table rule; the other two tagged misses are the box under the watermark and the box with handwriting.

On the external set it scored 47.5%, 199 of 419. It found only half the boxes (recall 51.8%), because those boxes are 12 to 31 px and often drawn as a `☐` character, outside the size and shape rules tuned on the 1004. Six of every ten boxes it returned were not annotated checkboxes (precision 39.2%), and `fill_ratio` called 17 empty boxes checked.

Its weakness is that its thresholds were set on those same 4 pages, so it only holds up on pages that look like them. A different form brings other box sizes and styles, as the external score shows. Before tuning, the samples dropped to 76.7% at half size, and on pages tilted a few degrees it kept 33.0% of the boxes (step 5).

### 3. Small pretrained models

I looked for models that run on CPU through onnxruntime, are small enough for a 1 to 2 GB container and return tight box coordinates. I evaluated the candidates on the samples.

Two of them failed on the samples. Docling's layout model draws one box around the checkbox and its label word, so at IoU 0.5 it matched 4% of the boxes. FFDetr found 44% of them and needed 1.47 GB of memory for a single page.

`form-field-v1-nano` (a 0.9M-parameter YOLOX-Nano) found 99.3% of the 288 sample boxes in 182 ms per page. `form-field-v1-state` (a 96k-parameter CNN) read the state of 99.7% of the 288 sample boxes and 98.2% of 2,728 crops from a public benchmark. Keeping locate and classify apart let me measure and swap each one alone. Out of the box, yolox + cnn scored 99.0% end-to-end on the 4 original samples and 57.3% on the external set.

### 4. Fine-tuning on public data

Four pages are too few to tune on, so I trained on synthetic pages built from the blank 1004, 1004C and 1004MC PDFs plus five CC BY 4.0 Roboflow datasets, and held out the 1004C as the validation form. I chose parameters on validation only and moved a value only when it gained at least half a point.

The external score rose from 57.3% to 71.8% while the 4 original samples stayed at 99.0%. Most of the gain came from YOLOX, which found 41 more boxes (recall from 70.4 to 80.2). The CNN cut checked boxes read as empty from 55 to 35.

The evaluation runs surfaced four errors along the way:

- The Roboflow boxes are 10 to 25% wider per side than the printed square. A YOLOX trained on them returned boxes about 30% too large, and the external score fell to 24.3%.
- After a first fine-tune, the CNN called 35 empty boxes with thick borders checked. About 21% of the checked training boxes had a colored tick that the grayscale step erases, so the model learned other cues for "checked". Leaving those boxes out of training took false checks on the external set from 35 to 0.

I made two choices after seeing evaluation data. I added the half-point rule after a smaller gain halved OpenCV's external recall. I also left `LabelParams` untuned, because the search raised the validation score but read the samples' labels worse.

### 5. Straighten the page first

A real upload is rarely perfectly straight. A page scanned by hand or photographed with a phone often comes in tilted a few degrees, and a scan can come in sideways. To measure that, I added the 4 originals tilted by 2 to 5° and turned 90°, plus 15 more filled appraisal reports, for 27 sample pages. The two locators failed in opposite places. OpenCV found only 33.0% of the boxes on the tilted pages, because its line removal and shape filters assume axis-aligned edges. YOLOX found 61.5% on the sideways ones. So I added a stage that straightens the page before anything else runs. With it, OpenCV + `fill_ratio` scores 94.5% end-to-end on the tilted pages and yolox + cnn 99.7% on the sideways ones.

So a straighten stage now runs before detection. RapidOCR finds the text lines, which decide quarter and half turns, a projection profile levels a tilt of up to ±10°, and the boxes are mapped back to the original image's pixels. With yolox + cnn the tilted pages now score 99.0% and the sideways ones 99.7%. Every request pays for it, at about 1.2 s per page.

### 6. The default: yolox + cnn

It is the best combination on both sets and makes no false checks on the external set. It also holds across scale, at 99.3% at 0.5×, 1× and 1.5×, while OpenCV loses 17 points at half size. OpenCV + `fill_ratio` stays as the baseline and as the fallback that needs no model files.

## Models and algorithms

This is the default pipeline, yolox + cnn, chosen after evaluating every combination (section 6). The classical baseline, opencv + fill_ratio, is described in section 2. Every stage works on a gray page, with colour overlays such as red watermarks and blue cell fills turned white first. The models run on onnxruntime.

### Straighten

RapidOCR's text detector finds the text lines on the page shrunk to 1,600 px. If most lines are taller than wide, the page is turned 90°. The tilt is then found by projection profile: the binarized page is rotated by every angle from -10° to +10° in steps of 0.25°, and the angle whose row sums have the highest variance wins, because level text lines give peaky sums. Last, the 10 widest lines are read as they are and turned 180°; if the turned copies score higher, the page is upside down. A page with fewer than 5 text lines is left alone. Boxes are mapped back to the original pixels through the inverse affine transform.

### Locate

YOLOX-Nano is a one-stage, anchor-free detector (depth 0.33, width 0.25, depthwise convolutions). The published checkpoint, `nutrientdocs/form-field-v1-nano`, has three classes: Text, ChoiceButton and Signature. The page is cut into tiles 0.6 page widths wide with 8% overlap, each letterboxed to 640×640 on the grey the model was trained with. The model returns 8,400 rows of [cx, cy, w, h, objectness, 3 class probabilities]; a row is a checkbox when objectness × P(ChoiceButton) ≥ 0.2 and ChoiceButton is the top class. Tiles are merged with NMS at IoU 0.3, and boxes under 8 px or over 6% of the page width are dropped.

Fine-tuning ran 20 epochs of 4,000 random tiles, SGD with Nesterov momentum and a OneCycle schedule at lr 0.002, EMA weights, and scan-like augmentation (contrast, blur, noise). The epoch with the best F2 on 200 validation pages was exported back to ONNX with the published input and output names, so the service only swaps the file.

### Classify

`nutrientdocs/form-field-v1-state` is a 96k-parameter network: a 3×3 convolution, five depthwise-separable blocks from 32 to 192 channels, global average pooling and a linear layer to two classes. The input is the box padded by 5% of its side, letterboxed to 40×160 gray and scaled to [-1, 1]; all crops of a page go in one batch. Softmax gives P(filled). The cut is 0.7 instead of the model's own 0.5, so an ambiguous crop comes out unchecked.

Fine-tuning ran 12 epochs with AdamW at lr 0.002 and class-weighted cross entropy. Augmentation adds contrast, brightness and noise, plus 1 to 3 specks in half of the empty crops; without them the model read any dot inside a box as checked. The epoch with the best balanced accuracy on validation was exported.

### Label

RapidOCR's recogniser (PP-OCR on onnxruntime) reads a strip right of each box: the box's row band, starting 5% of the side past the edge and ending at the next box on the row, a vertical rule, a blank gap of one box side after the text, or 15 box sides. Text with recognition confidence below 0.5 is dropped.

## Evaluation

The numbers come from `make eval` on an Apple M4 with one thread. Neither set was used for tuning.

- Samples: 27 pages and 1,370 boxes (`backend/tests/data/ground_truth/`). They are the 4 original pages, the same 4 tilted by 2 to 5° and turned 90°, and 15 more filled appraisal reports.
- External: 15 pages and 419 boxes from a public-domain Roboflow set of housing forms, labeled by third parties. They are not appraisal forms, the boxes are 12 to 31 px and most marks are blue pen (`backend/tests/data/external/README.md`).

A box counts only if it was found at IoU ≥ 0.5 with the right state. `test_golden.py` fails when a change loses more than one box on any sample page.

## Results

### End-to-end accuracy per combination

| Combination           | Samples | External | p50 ms/page (samples) |
| --------------------- | -------: | --------: | ---------------------: |
| opencv + fill\_ratio  | 96.9    | 49.9     | 1441                  |
| opencv + cnn          | 96.9    | 50.8     | 1301                  |
| yolox + fill\_ratio   | 98.8    | 70.4     | 1459                  |
| yolox + cnn (default) | 99.3    | 71.8     | 1378                  |

### Before and after tuning

"Before" is the pretrained models with the earlier parameters. Only the 4 original pages existed then.

| Combination          | 4 original pages, before | After | External, before | After |
| -------------------- | ------------------------: | -----: | ----------------: | -----: |
| opencv + fill\_ratio | 96.9                     | 96.9  | 47.5             | 49.9  |
| opencv + cnn         | 96.9                     | 96.9  | 50.8             | 50.8  |
| yolox + fill\_ratio  | 98.6                     | 99.0  | 45.6             | 70.4  |
| yolox + cnn          | 99.0                     | 99.0  | 57.3             | 71.8  |

## Alternatives considered

- Vision-language models (Qwen-VL, Gemini, Florence-2). They return coordinates on a coarse grid: Gemini's 0 to 1000 scale is 3.3 px per step on a 3,300 px page, about a quarter of a small box. They also downscale the page, which shrinks small boxes further, and take tens of seconds to minutes per page on CPU. On CheckboxQA, which asks only about state and needs no coordinates, the best one scored 83.2 against 97.5 for humans.
- Docling layout models (heron and egret). They have checkbox classes, but the 640 px input squashes the page and the box they return covers the checkbox and its label word. In a probe on the samples the state was often wrong too ("☒ Fee Simple" came back unselected). The heron ONNX file is 171 MB.
- FFDNet S and L. The ONNX metadata says AGPL-3.0. In a probe, S found mostly empty boxes and missed checked ones, and L found nothing on 3 of the 4 samples.
- `wendys-llc/checkbox-detector`. Its card says Apache-2.0 but its ONNX metadata says AGPL-3.0 (Ultralytics YOLO12n), and without its own tiling code it missed many checked boxes.
- Full-page OCR to find the labels. It took about 10 s and 2.5 GB per page on one thread, against about 9 ms per crop for recognition only.
- Other OCR engines. PaddleOCR 3.x pins an older NumPy and adds a 195 MB wheel, docTR needs PyTorch, and Tesseract needs system packages. RapidOCR runs the same PP-OCR models on onnxruntime from a pip wheel.
- Training a detector from scratch. A few thousand public pages are too little data for that, and fine-tuning a pretrained detector needs far less.

## Strengths and known limitations

What holds up:

- Pages tilted up to ±10°, on their side or upside down are turned upright first, and the boxes come back in the original image's pixels. The tilted samples score 99.0% and the sideways ones 99.7%.
- yolox + cnn scores 99.3% at 0.5×, 1× and 1.5× scale.
- It called no empty box checked on the external set, whose layouts it never saw.
- It needs no network, credentials or GPU, and `opencv + fill_ratio` still runs if the model files are missing.
- PDFs up to 100 pages are rendered one page at a time, so memory does not grow with the page count, and `/detect/stream` sends each page's boxes as soon as that page is done.

What does not:

- Orientation is read from RapidOCR text lines, so a page with fewer than 5 lines of text is not turned. It costs about 1.2 s per page on one thread, and every request pays it. Tilts beyond ±10° that are not a quarter turn, and perspective from phone photos, are not corrected.
- Round radio buttons are out of scope, since the labeling policy only counts printed squares and rectangles. Box styles that appear in neither the samples nor the training data, such as rounded corners or dashed borders, are untested and likely missed by `opencv`.
- Labels are read only to the right of the box. A box whose label sits on its left (the utilities table on the 1004) gets `null` or a stray glyph, and long sentence labels can come out garbled.
- The CNN reads a stroke that crosses an empty box as checked. `fill_ratio` handles that case but is less accurate on the external set overall.
- A request has no timeout and is not cancelled when the client disconnects: a 100-page PDF keeps a worker busy for about 2 minutes either way. The page cap is the only limit.

## Next steps

- Train and evaluate on real appraisal scans, which the public data lack.
- Read labels left of the box, and attach the question of the row the box answers.
- For fillable, born-digital PDFs, read the checkbox values straight from the AcroForm fields with pypdfium2. That gives exact answers with no vision.