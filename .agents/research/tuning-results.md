# Tuning on public data: results and error analysis

Ticket #21, done 2026-10-11/12. This note records the numbers and the error analysis behind the
fine-tuned `yolox` and `cnn` and the `*Params` chosen on validation, for the README write-up and the
default-pipeline decision (#22). How the data and training are built: `.agents/spec.md` §6, "Tuning on
public data", and `backend/training/`.

**Conclusion.** The fine-tuned YOLOX is the big gain: on unseen forms (external set) `yolox + cnn` goes
from 57.3% to 71.8% end-to-end, and the samples stay at 99.0%. Recommended default for #22: fine-tuned
`yolox` + fine-tuned `cnn`; `fill_ratio` stays as the classifier without a model, and `opencv +
fill_ratio` as the baseline. The service default is still `opencv + fill_ratio` until #22.

## How it was measured

- **Validation** (to choose): 610 pages held out from training, 10,426 boxes. The synthetic 1004C pages
  plus real Roboflow scans in whole layout groups. Checkpoints and parameters were chosen here only.
- **Evaluation** (to judge, never tuned on): the 4 samples (288 boxes) and the external set (15 pages,
  419 boxes, labelled by third parties; `backend/tests/data/external/README.md`).
- **End-to-end accuracy** (headline): a true box counts only if found at IoU ≥ 0.5 **and** its state is
  right. Tables from `backend/scripts/evaluate.py`. "Before" = pretrained models and the earlier params
  (run on `71d3e68`); "after" = commit `00b468e`.

## Before / after, end-to-end

| combo | samples before | samples after | external before | external after |
|---|--:|--:|--:|--:|
| opencv + fill_ratio | 96.9 | 96.9 | 47.5 | 49.9 |
| opencv + cnn | 96.9 | 96.9 | 50.8 | 50.8 |
| yolox + fill_ratio | 98.6 | 99.0 | 45.6 | 70.4 |
| **yolox + cnn** | 99.0 | **99.0** | 57.3 | **71.8** |

### Detail, totals per combo

P / R = detection precision and recall at IoU 0.5; state = state accuracy on matched boxes;
false ✓ = empty box called checked; false ☐ = checked box called empty.

| set | combo | P | R | R@0.3 | state | false ✓ | false ☐ | end-to-end |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| samples, before | opencv + fill_ratio | 98.9 | 96.9 | 96.9 | 100.0 | 0 | 0 | 96.9 |
| samples, after | opencv + fill_ratio | 82.3 | 96.9 | 96.9 | 100.0 | 0 | 0 | 96.9 |
| samples, before | opencv + cnn | 98.9 | 96.9 | 96.9 | 100.0 | 0 | 0 | 96.9 |
| samples, after | opencv + cnn | 82.3 | 96.9 | 96.9 | 100.0 | 0 | 0 | 96.9 |
| samples, before | yolox + fill_ratio | 95.3 | 99.3 | 99.3 | 99.3 | 2 | 0 | 98.6 |
| samples, after | yolox + fill_ratio | 94.7 | 99.3 | 100.0 | 99.7 | 1 | 0 | 99.0 |
| samples, before | yolox + cnn | 95.3 | 99.3 | 99.3 | 99.7 | 1 | 0 | 99.0 |
| samples, after | yolox + cnn | 94.7 | 99.3 | 100.0 | 99.7 | 1 | 0 | 99.0 |
| external, before | opencv + fill_ratio | 39.2 | 51.8 | 52.0 | 91.7 | 17 | 1 | 47.5 |
| external, after | opencv + fill_ratio | 66.3 | 51.6 | 51.6 | 96.8 | 4 | 3 | 49.9 |
| external, before | opencv + cnn | 39.2 | 51.8 | 52.0 | 98.2 | 0 | 4 | 50.8 |
| external, after | opencv + cnn | 66.3 | 51.6 | 51.6 | 98.6 | 0 | 3 | 50.8 |
| external, before | yolox + fill_ratio | 72.7 | 70.4 | 70.6 | 64.7 | 88 | 16 | 45.6 |
| external, after | yolox + fill_ratio | 66.3 | 80.2 | 82.8 | 87.8 | 11 | 30 | 70.4 |
| external, before | yolox + cnn | 72.7 | 70.4 | 70.6 | 81.4 | 0 | 55 | 57.3 |
| external, after | yolox + cnn | 66.3 | 80.2 | 82.8 | 89.6 | 0 | 35 | 71.8 |

### By scale (samples, end-to-end)

| combo | before 0.5x | before 1x | before 1.5x | after 0.5x | after 1x | after 1.5x |
|---|--:|--:|--:|--:|--:|--:|
| opencv + fill_ratio | 76.7 | 96.9 | 96.9 | 76.0 | 96.9 | 96.9 |
| opencv + cnn | 76.4 | 96.9 | 96.9 | 76.0 | 96.9 | 96.9 |
| yolox + fill_ratio | 99.3 | 98.6 | 99.0 | 99.3 | 99.0 | 99.3 |
| yolox + cnn | 98.6 | 99.0 | 98.6 | 99.7 | 99.0 | 99.3 |

### By tag (samples, end-to-end)

| combo | checkmark (n=1) | color_fill (n=18) | handwriting (n=1) | touches_line (n=72) | watermark (n=1) |
|---|--:|--:|--:|--:|--:|
| opencv + fill_ratio, before → after | 100 → 100 | 100 → 100 | 0 → 0 | 93.1 → 93.1 | 0 → 0 |
| opencv + cnn, before → after | 100 → 100 | 100 → 100 | 0 → 0 | 93.1 → 93.1 | 0 → 0 |
| yolox + fill_ratio, before → after | 100 → 100 | 100 → 100 | 0 → 0 | 97.2 → 100 | 100 → 100 |
| yolox + cnn, before → after | 100 → 100 | 100 → 100 | 0 → 0 | 97.2 → 100 | 100 → 100 |

### Label accuracy (samples, matched boxes with a ground-truth label)

| combo | before | after |
|---|--:|--:|
| opencv + fill_ratio / cnn | 84.8 (244 labels) | 79.5 (244) |
| yolox + fill_ratio / cnn | 82.7 (249) | 81.7 (251) |

`LabelParams` did not change; the opencv drop comes from its extra false-positive boxes, which end the
label strip early (the strip stops at the next box on the row).

Latency is not compared: the "after" run was made while the machine was loaded (every combo, including
OpenCV, which did not change speed, measured about twice the "before" ms). Re-measure before the README.

## Validation, per stage

| Stage | What | Before | After |
|---|---|--:|--:|
| `yolox` | detection F2 at IoU 0.5 (200 pages) | 0.779 | **0.942** (recall 0.72 → 0.95) |
| `cnn` | state accuracy, true boxes | 0.955 (pretrained, best setting) | **0.982** |
| `OpenCVParams` | detection F2 (300 pages) | 0.748 | 0.810 |
| `FillParams` | state accuracy, best threshold per try | 0.908 | 0.930 |
| `BinarizeParams`, `YoloxParams`, `StateCnnParams` | no field gained 0.5 points | — | unchanged |
| `LabelParams` | not tuned (see below) | — | unchanged |

YOLOX tile size: the pretrained model scored F2 0.52–0.89 over `tile_frac` 0.5–1.0 and score threshold
0.1–0.3; the fine-tuned one 0.930–0.946, so one tile size is enough.

## Errors corrected, and how they were found

1. **YOLOX missed boxes on scanned forms.** The pretrained model only saw boxes at a narrow share of the
   tile. Fine-tuned on tiles cut the way the service cuts them, it finds 41 more of the 419 external
   boxes (recall 70.4 → 80.2) and no longer depends on the tile size.
2. **The pretrained CNN missed hand-drawn marks on real scans** (pen ticks and crosses): checked boxes
   read as empty on the external set went from 55 to 35.
3. **Loose training boxes.** Roboflow annotations are 10–25% wider per side than the printed square
   (median share of the side filled by the square: 0.75–0.91 by source, against 0.95 on exact synthetic
   boxes). A YOLOX trained on them returned boxes ~30% too large: external end-to-end fell to 24.3 and
   recall at IoU 0.5 to 27.4, while recall at IoU 0.3 was 86.4. Fix: `data.py` snaps each annotation to
   the printed square inside it.
4. **CNN called thick-bordered empty boxes checked.** After a first fine-tune, 35 external boxes, all
   empty ☐ glyphs with a thick, broken border, scored P(filled) 0.72–1.00 (the pretrained model: 0.00–0.05;
   `fill_ratio`: 0 ink). Cause: about 21% of the checked training boxes had a colour tick that
   `to_clean_gray` erases, so the model saw an empty box labelled checked and learned other cues, such as
   a thick border (by source: checkboxes-2 29%, government-omr 21%, synthetic 17%). Fix: those boxes are
   left out of the CNN crops. False checked on the external set: 35 → 0.
5. **CNN called specks checked.** A 2 px dot in a 60 px box scored 0.92. Fix: dust specks drawn into
   empty training crops (labelled unchecked); after it, only fills of about a third of the side read
   checked.
6. **`fill_ratio` counted a stray stroke as a mark** (bug, commit `6729d29`). A stroke crossing the box
   edge was cut where the edge was removed; the closing meant to rejoin it fails at a shallow angle, so the
   inside piece counted as a mark wholly inside the box. The stray-stroke test passed only because that
   piece measured 0.061, under the 0.07 threshold. Fixed with a dilation; the test now asserts 0 ink.

## What did not improve, or got worse

- **OpenCV**: recall unchanged on both sets; precision on the samples 98.9 → 82.3 (the search maximises
  F2, which weighs recall 4x). Precision on the external set rose 39.2 → 66.3.
- **Labels on the samples**: see the table above.
- **YOLOX precision on the external set** 72.7 → 66.3. The external README notes printed boxes that the
  third party did not annotate, so part of these "false positives" are real boxes.
- **Remaining external errors of `yolox + cnn`**: 35 checked boxes read as empty. They are large blue pen
  strokes drawn over a small box, mostly outside it (ticks sweeping over the box and the word, circles
  around them). The labeling policy (spec §7) counts only marks mostly inside the box, so this is a
  policy difference with that dataset; a few X marks over the box score 0.5–0.65, just under the 0.7
  threshold. 9 of the 35 are colour marks that `to_clean_gray` erases.
- **Known CNN limit, unchanged**: a stray stroke crossing an empty box reads as checked (`fill_ratio`
  handles it).

## Choices informed by evaluation data (disclosed)

The rule is that parameters are chosen on validation only. Two choices were shaped by evaluation data
after seeing it, with the user's approval:

- **0.5-point minimum gain** for moving a parameter. Added after a first search moved `max_aspect`
  1.4 → 1.25 for 0.1–0.4 points of F2, which halved OpenCV recall on the external set (its boxes are
  taller ☐ glyphs). It applies to every stage alike.
- **`LabelParams` not applied.** A search on synthetic 1004C pages (labels from the PDF text layer) moved
  `band_margin_frac` 0.1, `start_gap_frac` 0.15, `ink_below` 160 and raised validation exact match
  0.765 → 0.876, but read the samples' labels 7–11 points worse (84.8 → 73.4 with opencv, 82.7 → 75.7 with
  yolox). Likely, but not verified: the text-layer ground truth cuts at line breaks and sometimes runs
  across columns, unlike the typed sample labels.

The CNN fix (error 4) was found by looking at external errors, but changes only which training boxes are
used (a data-quality rule), not a parameter.

Also kept against validation: `fill_threshold` 0.07, where 0.02 (the bottom of the tried range) was 1.6
points more accurate; a 2% speck would then read as checked (spec: ambiguous resolves to unchecked).

## Data and licences

| Source | Licence | Used for | Pages in build (train + val) |
|---|---|---|--:|
| Roboflow `cogniquest-idc1w/checkboxes-xtv3p` (fork) | CC BY 4.0 | boxes only (its `checked` class is mostly empty boxes) | 2,111 |
| Roboflow `myproject-gemsb/checkboxes-2-dyp1n` (fork) | CC BY 4.0 | boxes + state, 6 class names remapped | 1,137 |
| Roboflow `tempqqaq/checkbox_v2` (fork) | CC BY 4.0 | boxes + state | 193 |
| Roboflow `ujiboo/checkbox-detector-zxwoh` (fork) | CC BY 4.0 | boxes + state, web screenshots dropped | 119 |
| Roboflow `s-workspace-dte1y/government-document-omr-checkbox` (fork) | CC BY 4.0 | boxes + state | 155 |
| Blank 1004 (Freddie Mac form 70), 1004C (70B), 1004MC (Fannie Mae) | government-sponsored forms, freely downloadable | synthetic pages; 1004C is the validation form | 50 + 50 |
| Dropped: `tax-form-checkbox-detection` (radio buttons with their word), `checkbox-detection-jiid0` (10 px crops) | CC BY 4.0 | — | 0 |
| Excluded: `annotation-mmvj7/new_merge_project` (the external set), AutoFormBench, CheckboxQA | — | never read by `training/` | 0 |

Totals: train 3,205 pages / 48,414 boxes; validation 610 pages / 10,426 boxes. Pretrained weights:
`nutrientdocs/form-field-v1-nano` and `form-field-v1-state` (Apache-2.0).

## Tests

**Written or changed:**

| Test | Level | The edit that breaks it |
|---|---|---|
| `test_golden.py::test_end_to_end_accuracy_stays_above_floor` (4 combos × 4 samples) | Integration | Any stage change that loses more than one box on a sample (floor = accuracy after tuning minus one box) |
| `test_states_per_mark_class`: stray stroke adds 0 ink | Unit | Going back to the closing in `mark_ratio`: the stray stroke at 90 px reads checked (checked by breaking it on purpose) |
| `test_near_threshold_fill_is_unchecked`: 9×9 dot | Unit | Lowering `fill_threshold` under ~0.06, or the interior margin changing so the dot leaves the ambiguous band |
| `test_cnn_ambiguous_score_is_unchecked`: 16×19 dot, P ≈ 0.65 | Unit | Lowering `StateCnnParams.threshold` under 0.65 |

**Skipped:**

| Skip | Why |
|---|---|
| Tests for `backend/training/` | Offline scripts, not part of the service; their output is checked by validation metrics and the golden tests |
| One test per changed `*Params` value | Locks constants; behaviour is covered by the golden and stage tests |
| ONNX input/output shapes of the tuned models | Every locate/classify/API test loads and runs them |
