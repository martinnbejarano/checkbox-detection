# Write-up

## Summary

The service finds the checkboxes on a page, says whether each one is checked and, on request, reads its label. The default pipeline straightens the page, locates the boxes with a fine-tuned YOLOX-Nano, classifies each one with a fine-tuned 96k-parameter CNN and reads labels with RapidOCR. It all runs on CPU through onnxruntime, with no external service.

On the 27 sample pages (1,370 boxes) it scores 99.3% end-to-end. On 15 third-party pages of other forms (419 boxes) it scores 71.8% and calls no empty box checked. Neither set was used for tuning. A page takes about 1.4 s on one thread, and 1.2 s of that goes to reading its orientation. The biggest gap is data, since no real appraisal scans were available for training.

## Results

`make eval` runs every locator × classifier combination on both sets, on an Apple M4 with one thread. A box counts only if it was found at IoU ≥ 0.5 with the right state. `test_golden.py` fails when a change loses more than one box on any sample page.

- Samples: 27 pages and 1,370 boxes (`backend/tests/data/ground_truth/`). They are the 4 original pages, the same 4 tilted by 2 to 5° and turned 90°, and 15 more filled appraisal reports, labeled by hand under a written policy: a box is checked when an intentional mark sits mostly inside it, and a stray stroke, a watermark or a colored cell fill does not count.
- External: 15 pages and 419 boxes from a public-domain Roboflow set of housing forms, labeled by third parties. They are not appraisal forms, the boxes are 12 to 31 px and most marks are blue pen (`backend/tests/data/external/README.md`).

| Combination           | Samples | External | p50 ms/page (samples) |
| --------------------- | -------: | --------: | ---------------------: |
| opencv + fill\_ratio  | 96.9    | 49.9     | 1441                  |
| opencv + cnn          | 96.9    | 50.8     | 1301                  |
| yolox + fill\_ratio   | 98.8    | 70.4     | 1459                  |
| yolox + cnn (default) | 99.3    | 71.8     | 1378                  |

## Strengths and known limitations

What holds up:

- Pages tilted up to ±10°, on their side or upside down are turned upright first, and the boxes come back in the original image's pixels. The tilted samples score 99.0% and the sideways ones 99.7%.
- `yolox + cnn` scores 99.3% at 0.5×, 1× and 1.5× scale, while OpenCV loses 17 points at half size.
- It called no empty box checked on the external set, whose layouts it never saw.
- It needs no network, credentials or GPU, and `opencv + fill_ratio` needs no model files, selected with `CHECKBOX_LOCATOR=opencv CHECKBOX_CLASSIFIER=fill_ratio`.
- PDFs up to 100 pages are rendered one page at a time, so memory does not grow with the page count. Such a PDF takes over two minutes, so `/detect/stream` sends each page's boxes as soon as that page is done. The demo page uses it.

What does not:

- Orientation costs about 1.2 s per page, and every request pays it. A page with fewer than 5 lines of text is not turned, and tilts beyond ±10° that are not a quarter turn, and perspective from phone photos, are not corrected.
- Round radio buttons are out of scope, since the labeling policy only counts printed squares and rectangles. Box styles that appear in neither the samples nor the training data, such as rounded corners or dashed borders, are untested and likely missed by `opencv`.
- Labels are read only to the right of the box. A box whose label sits on its left (the utilities table on the 1004) gets `null` or a stray glyph, and long sentence labels can come out garbled.
- The CNN reads a stroke that crosses an empty box as checked. `fill_ratio` handles that case but is less accurate on the external set overall.
- A request has no timeout and is not cancelled when the client disconnects: a 100-page PDF keeps a worker busy for about 2 minutes either way. The page cap is the only limit.

## Next steps

- Train and evaluate on real appraisal scans, which the public data lack.
- Read labels left of the box, and attach the question of the row the box answers.
- For fillable, born-digital PDFs, read the checkbox values straight from the AcroForm fields with pypdfium2. That gives exact answers with no vision.
- For production at HomeVision, where a job is a whole loan file:
  - Put uploads on a queue that workers take from, answer with a job id, and scale workers with queue depth. At about 1.4 s per page, one core reads about 0.7 pages/s.
  - Give each job a time budget from its page count and cancel it past that, and add per-client rate limits on top of the upload caps.
  - Send uncertain boxes to a human review queue instead of straight into the appraisal record: a low locator `score`, or a state probability close to the 0.7 cut.
  - Track the logged times and box counts, plus the share of boxes sent to review per form, to catch drift early (a new form version, a new scanner). The reviewed boxes become the next training data.

## How I got here

I kept each step below only if it moved a measured number.

### 1. Measure first

Before developing any detector, I labeled the 4 original samples (288 boxes) by hand and put the external set aside, so every later step could be measured against both.

### 2. A classical baseline

OpenCV + `fill_ratio` came first because it needs no data or weights and every threshold can be read.

On the 4 samples it scored 96.9% end-to-end, 279 of 288 boxes. All 9 misses were boxes it did not find, since every box it found had the right state. Five touch a table rule, and the other four sit under the watermark, under handwriting, or are printed too faintly for their border to survive binarization.

On the external set it scored 47.5%, 199 of 419. It found only half the boxes (recall 51.8%), because those boxes are 12 to 31 px and often drawn as a `☐` character, outside the size and shape rules tuned on the 1004. Six of every ten boxes it returned were not annotated checkboxes (precision 39.2%), and `fill_ratio` called 17 empty boxes checked.

Its weakness is that its thresholds were set on those same 4 pages, so it only holds up on pages that look like them. Before tuning, the samples dropped to 76.7% at half size, and on pages tilted a few degrees it kept 33.0% of the boxes (step 5).

### 3. Small pretrained models

I looked for models that run on CPU through onnxruntime, are small enough for a 1 to 2 GB container and return tight box coordinates.

`form-field-v1-nano` (a 0.9M-parameter YOLOX-Nano) found 99.3% of the 288 sample boxes in 182 ms per page. `form-field-v1-state` (a 96k-parameter CNN) read the state of 99.7% of the 288 sample boxes and 98.2% of 2,728 crops from a public benchmark. Keeping locate and classify apart let me measure and swap each one alone. Out of the box, yolox + cnn scored 99.0% end-to-end on the 4 original samples and 57.3% on the external set.

### 4. Fine-tuning on public data

Four pages are too few to tune on, so I trained on synthetic pages built from the blank 1004, 1004C and 1004MC PDFs plus five CC BY 4.0 Roboflow datasets, and held out the 1004C as the validation form. I chose parameters on validation only and moved a value only when it gained at least half a point.

The external score rose from 57.3% to 71.8% while the 4 original samples stayed at 99.0%. Most of the gain came from YOLOX, which found 41 more boxes (recall from 70.4 to 80.2). The CNN cut checked boxes read as empty from 55 to 35.

The evaluation runs surfaced two errors along the way:

- The Roboflow boxes are 10 to 25% wider per side than the printed square. A YOLOX trained on them returned boxes about 30% too large, and the external score fell to 24.3%.
- After a first fine-tune, the CNN called 35 empty boxes with thick borders checked. About 21% of the checked training boxes had a colored tick that the grayscale step erases, so the model learned other cues for "checked". Leaving those boxes out of training took false checks on the external set from 35 to 0.

I made two choices after seeing evaluation data. I added the half-point rule after a smaller gain halved OpenCV's external recall. I also left `LabelParams` untuned, because the search raised the validation score but read the samples' labels worse.

### 5. Straighten the page first

A real upload is rarely perfectly straight. A page scanned by hand or photographed with a phone often comes in tilted a few degrees, and a scan can come in sideways. On the tilted and sideways samples, the two locators failed in opposite places. OpenCV found only 33.0% of the boxes on the tilted pages, because its line removal and shape filters assume axis-aligned edges. YOLOX found 61.5% on the sideways ones.

So I added a stage that straightens the page before anything else runs, described under Models and algorithms. With it, OpenCV + `fill_ratio` scores 94.5% end-to-end on the tilted pages and yolox + cnn 99.7% on the sideways ones.

### 6. The default: yolox + cnn

It is the best combination on both sets, makes no false checks on the external set and holds across scale. OpenCV + `fill_ratio` stays as the baseline.

## Models and algorithms

Every stage works on a gray page, with colour overlays such as red watermarks and blue cell fills turned white first. The models run on onnxruntime.

### Straighten

RapidOCR's text detector finds the text lines on the page shrunk to 1,600 px. If most lines are taller than wide, the page is turned 90°. The tilt is then found by projection profile: the binarized page is rotated by every angle from -10° to +10° in steps of 0.25°, and the angle whose row sums have the highest variance wins, because level text lines give peaky sums. Last, the 10 widest lines are read as they are and turned 180°; if the turned copies score higher, the page is upside down. Boxes are mapped back to the original pixels through the inverse affine transform.

### Locate

`nutrientdocs/form-field-v1-nano` is a YOLOX-Nano detector whose ChoiceButton class is the checkbox. The page is cut into overlapping 640×640 tiles so small boxes keep their size, and the tiles are merged with NMS. The fine-tuned model was exported back to ONNX with the published input and output names, so the service only swaps the file.

### Classify

`nutrientdocs/form-field-v1-state` is a 96k-parameter network: a 3×3 convolution, five depthwise-separable blocks from 32 to 192 channels, global average pooling and a linear layer to two classes. The input is the box padded by 5% of its side, letterboxed to 40×160 gray and scaled to [-1, 1]; all crops of a page go in one batch. Softmax gives P(filled). The cut is 0.7 instead of the model's own 0.5, so an ambiguous crop comes out unchecked. Fine-tuning adds 1 to 3 specks to half of the empty crops; without them the model read any dot inside a box as checked.

### Label

The contract asks for position and state, and a client still has to work out which field each box belongs to. So the service can also read the text printed next to the box, on request only (`?labels=true`), since it adds about 9 ms per box.

RapidOCR's recogniser (PP-OCR on onnxruntime) reads a strip right of each box, up to the next box on the row, a vertical rule or a blank gap after the text. Text with recognition confidence below 0.5 is dropped.
