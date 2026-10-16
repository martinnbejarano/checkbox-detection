# Write-up

## Summary

The service finds the checkboxes on a page, says whether each one is checked and, on request, reads its label. A page is straightened first, then a fine-tuned YOLOX-Nano locates the boxes, a fine-tuned 96k-parameter CNN classifies each one and RapidOCR reads the labels. Everything runs on CPU through onnxruntime.

On the 27 sample pages (1,370 boxes) it scores 99.1% end-to-end. On 15 third-party pages of other forms (419 boxes) it scores 79.2% and calls no empty box checked. Neither set was used for tuning. A page takes about 1.4 s on one thread, 1.2 s of it reading the orientation. The biggest gap is data: no real appraisal scans were available for training.

## Results

`make eval` runs every locator × classifier combination on both evaluation sets (Apple M4, one thread). A box counts only if it was found at IoU ≥ 0.5 with the right state.

- Samples: the 4 original pages, the same 4 tilted by 2 to 5° and turned 90°, and 15 more filled appraisal reports, labeled by hand. A box is checked when an intentional mark sits mostly inside it; a stray stroke, a watermark or a colored cell fill does not count.
- External: 15 public-domain housing-form pages from Roboflow, labeled by third parties. Other layouts, boxes of 12 to 31 px, most marks in blue pen. Never used for tuning.
- Housing validation: 71 other pages of the same Roboflow set (2,823 boxes), used to choose parameters.

| Combination           | Samples | External | Housing val | p50 ms/page (samples) |
| --------------------- | -------: | --------: | ----------: | ---------------------: |
| opencv + fill\_ratio  | 96.9    | 49.9     | 54.1       | 1399                  |
| opencv + cnn          | 96.9    | 50.8     | 54.3       | 1388                  |
| yolox + fill\_ratio   | 99.3    | 70.2     | 70.8       | 1430                  |
| yolox + cnn (default) | 99.1    | 79.2     | 77.4       | 1351                  |

## Strengths and known limitations

What holds up:

- Pages tilted up to ±10°, on their side or upside down are turned upright first, and the boxes come back in the original pixels. Both tilted and sideways samples score 98.6%.
- `yolox + cnn` holds across scale (99.3%, 99.1% and 99.3% at 0.5×, 1× and 1.5×) and called no empty box checked on layouts it never saw.
- PDFs up to 100 pages are rendered one page at a time, and `/detect/stream` sends each page's boxes as soon as that page is done.

What does not:

- Orientation costs about 1.2 s per page, on every request. Tilts beyond ±10° that are not a quarter turn, and perspective from phone photos, are not corrected.
- Round radio buttons are out of scope. Rounded or dashed borders appear in neither the samples nor the training data and are untested.
- Labels are read only to the right of the box. A label on the left gets `null` or a stray glyph, and long sentence labels can come out garbled.
- On the external set the locator misses 18% of the boxes, nearly all checked ones under a heavy pen stroke; the boxes it finds get the right state 97% of the time. The CNN reads a stroke that crosses an empty box as checked.
- A mark beside the box (a circle around it, a tick on a corner) is unchecked under the labeling policy; the third-party labels count it checked.
- A request has no timeout and is not cancelled when the client disconnects. The page cap is the only limit.

## Next steps

- Train and evaluate on real appraisal scans, which the public data lack. Synthetic marks barely moved the locator's recall.
- Read labels left of the box, and attach the question of the row the box answers.
- For born-digital PDFs, read the checkbox values from the AcroForm fields with pypdfium2.
- For production at HomeVision, where a job is a whole loan file: a queue with workers scaled on its depth, a time budget per job, rate limits per client, and a human review queue for boxes with a low `score` or a state probability near the 0.7 cut. The reviewed boxes become the next training data.

## How I got here

I kept each step only if it moved a measured number.

1. **Measure first.** Before any detector, I labeled the 4 original samples (288 boxes) by hand and put the external set aside.

2. **A classical baseline.** OpenCV + `fill_ratio` came first because it needs no data or weights and every threshold can be read. It scored 96.9% on the 4 samples and 47.5% on the external set, whose boxes are smaller, often a `☐` glyph, and fall outside size and shape rules tuned on 4 pages.

3. **Small pretrained models.** `form-field-v1-nano` found 99.3% of the sample boxes in 182 ms per page, and `form-field-v1-state` read the state of 99.7% of them, both on CPU through onnxruntime. Keeping locate and classify apart let me measure and swap each one alone. Out of the box, yolox + cnn scored 99.0% on the samples and 57.3% on the external set.

4. **Fine-tuning on public data.** I trained on synthetic pages built from the blank 1004, 1004C and 1004MC PDFs plus five CC BY 4.0 Roboflow datasets, held out the 1004C for validation, and moved a value only when it gained at least half a point there. External rose from 57.3% to 71.8% with the samples unchanged. Two errors surfaced on the way: the Roboflow boxes are 10 to 25% wider than the printed square, and a YOLOX trained on them returned boxes 30% too large; and 21% of the checked training crops had a colored tick that the grayscale step erases, so the CNN learned to call thick-bordered empty boxes checked. Leaving those crops out took false checks from 35 to 0.

5. **Straighten the page first.** On the tilted and sideways samples the two locators failed in opposite places: OpenCV found 33.0% of the boxes on tilted pages, YOLOX 61.5% on sideways ones. Straightening the page before anything else runs took them to 94.5% and 99.7%.

6. **Ticks that spill past the box.** The external set still read checked boxes at about 27%: a hand tick on a 17 px box is 2 to 3 sides long and mostly outside it, so a crop 5% past the box never saw the mark. The CNN now crops a third of a side past the box and reads plain gray, and both models were retrained on synthetic pages with big spilling marks and blue pen. External went from 71.8% to 79.2%, with checked boxes read as empty down from 35 to 10 and still no false checks.

`yolox + cnn` is the best combination on every set, so it is the default. OpenCV + `fill_ratio` stays as the baseline.

## Models and algorithms

Every stage but classify works on a gray page with colour overlays (red watermarks, blue cell fills) turned white first; the classifier gets plain gray, since a pen tick is saturated too.

**Straighten.** RapidOCR's text detector finds the text lines on the page shrunk to 1,600 px; if most lines are taller than wide, the page is turned 90°. The tilt is found by projection profile: the binarized page is rotated from -10° to +10° in steps of 0.25°, and the angle whose row sums have the highest variance wins, since level text lines give peaky sums. Then the 10 widest lines are read as they are and turned 180°; if the turned copies score higher, the page is upside down. Boxes are mapped back through the inverse affine transform.

**Locate.** `nutrientdocs/form-field-v1-nano`, a 0.9M-parameter YOLOX-Nano whose ChoiceButton class is the checkbox. The page is cut into overlapping 640×640 tiles so small boxes keep their size, and the tiles are merged with NMS.

**Classify.** `nutrientdocs/form-field-v1-state`, a 96k-parameter CNN with two classes. The input is the box padded by 30% of its side, letterboxed to 40×160 gray; all crops of a page go in one batch. The cut on P(filled) is 0.7 instead of 0.5, so an ambiguous crop comes out unchecked. Fine-tuning draws pads from 0 to 50% and adds specks to half of the empty crops; without them the model read any dot inside a box as checked.

**Label.** On request only (`?labels=true`, about 9 ms per box), RapidOCR's recogniser reads a strip right of each box, up to the next box on the row, a vertical rule or a blank gap after the text. Text with confidence below 0.5 is dropped.
