# External evaluation set (never used for tuning)

15 scanned and typed US housing forms (residency application, student eligibility declaration), 419 boxes
(155 checked). Their layouts and box sizes differ from `samples/`: boxes are 12–31 px on pages 1275–2550 px
wide, often drawn as a `☐` text glyph, and most marks are hand-drawn in blue pen.

## Source and license

- Roboflow Universe [`annotation-mmvj7/new_merge_project`](https://universe.roboflow.com/annotation-mmvj7/new_merge_project),
  861 images, **Public Domain**. Exported as COCO on 2026-10-11 from a fork of the project, because the public
  export links returned no archive. The fork keeps each page's original resolution; the published versions
  stretch every page to 640×640.
- Pages are stored as Roboflow exported them: JPEG re-encodes of the uploaded PNG/JPG files, same pixel size.
- Class remap: `checked` → `is_checked: true`, `unchecked` → `false`. Pages with any other class were not used.
  There is no `label` (the source has no text) and no `tags`.

## Selection rule

1. Only full-resolution pages. The 640×640 copies are stretched and dropped.
2. Only pages whose boxes are printed squares. The child-support declarations (`CSD-*`, half of `SKM_*`) mark
   answers by circling a word. That is out of scope: the labeling policy only counts printed squares.
3. Only pages with at least one checked and one unchecked box. Pages are deduplicated by (file name, size).
4. Pages are grouped by layout and image size, and the first pages of each group are taken in natural file-name
   order:

| Files | Group | Source file | Size | Boxes (checked) |
|---|---|---|---|---|
| external-00 | application p.1, scan | `1.png` | 2480×3505 | 75 (32) |
| external-01–04 | application p.2, scan | `3.png`, `4.png`, `5.png`, `7.png` | 2480×3505 | 6 (2–3) each |
| external-05 | application p.1, typed | `1.png` | 2550×4200 | 77 (16) |
| external-06–07 | application p.2, typed | `2.png`, `4.png` | 2550×4200 | 4 (2) each |
| external-08–09 | application p.1, cropped scan | `47.png`, `48.png` | 2031×3505, 2086×3235 | 75 (31), 75 (34) |
| external-10–11 | student eligibility, scan | `SKM_C25813010104090-52.png`, `-53.png` | 2479×3504 | 17 (12), 17 (3) |
| external-12–14 | student eligibility, typed | `SED-1-.png`, `SED-2-.png`, `SED-3-.png` | 1275×1650 | 17 (4) each |

The annual recertification pages were left out: their boxes are faint dotted glyphs that could not be located
reliably (step below).

## What was changed: box geometry only

The source annotates each box **together with its label text** (`☐ Part Time`; median width/height 2.6–4.5).
Matched against tight boxes at IoU ≥ 0.5 it would count almost every detection as wrong. So each region was
shrunk to the printed square at its left end:

- Template match of a hollow square of the page's box size, on the page with pen colour removed and only straight
  strokes kept, taking the leftmost strong match inside the region.
- Every box was then reviewed on zoomed crops. 49 of 419 had landed on a letter or a pen stroke; their corners
  were read by hand from a pixel grid.

Which boxes exist and whether each is checked is exactly as the third party labelled it. Known gaps in those
labels: some printed boxes are not annotated (for example the "Office use only" boxes on the application), so a
detector that finds them is charged a false positive.
