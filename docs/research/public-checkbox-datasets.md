# Public labeled checkbox datasets for forms

Resolves issue #2. Researched 2026-10-07 against dataset cards, papers, repos and the forms themselves.
It extends `deep-learning-and-data.md` §3/§8 and does not repeat it (CommonForms, FFDNet, the small Roboflow
sets, CheckboxQA and wendys-llc are covered there).

**Short answer.** No public dataset pairs *scanned US mortgage/appraisal forms* with checkbox boxes and checked
state. The best "real data" is a handful of small CC BY / public-domain Roboflow sets of scanned forms. The best
source overall is the **official blank forms**: the Fannie Mae 1004 PDF is a fillable AcroForm, and all three forms
draw every checkbox as an exact vector square. That gives pixel-exact synthetic labels for the actual templates in
`samples/` (which are 2550x4200 = 8.5x14 in legal at 300 DPI, the same page size as the PDFs).

## Ranked shortlist (for our use: scanned appraisal forms, bbox + is_checked)

| # | Source | Size | Labels / classes | Checked state | Scan? | License | Use |
|---|---|---|---|---|---|---|---|
| 1 | **Official blank 1004 / 1004MC / 1004C PDFs** (Fannie Mae / Freddie Mac) | 3 forms, 7+2+7 pp; ~200 boxes on 1004 pp.1-3, 51 on 1004MC p.1, ~350 on 1004C | Vector squares (8.1 pt, 7.2 pt, 8.7 pt); 1004 also has 118 CheckBox/Radio widgets on p.1 | We choose it (synthetic) | Born-digital → we degrade | Government-sponsored form, freely downloadable | **Primary training data via synthesis** |
| 2 | Roboflow `annotation-mmvj7/new_merge_project` | 861 imgs | 7 classes: `checked`, `unchecked`, 3 `checkbox-*` variants, 2 junk (`'`, `\`) | Yes | **Scanned US housing forms** (annual recertification, handwriting) | **Public Domain** | Best real-domain extra data; remap classes |
| 3 | Roboflow `cogniquest-idc1w/checkboxes-xtv3p` | 2,830 imgs | `checked`, `unchecked` | Yes | Scanned document pages (spot-checked 2 thumbs) | CC BY 4.0 | Largest real checked/unchecked set; pretrain + classifier crops |
| 4 | **NAF** (National Archives Forms, BYU) | 860 imgs (708/75/77) | Polygons; `fieldCheckBox` type + `isBlank` (3 = blank, 1 hand, 2 print) | **Derivable** (`isBlank`) | **Scanned historical US forms** | CDLA-Permissive-1.0 | Real scans with state; old forms, verify `isBlank` on boxes |
| 5 | `nutrientdocs/form-field-vlm-v2-benchmark` | 353 pages, 8,216 fields, **2,026 checkboxes** + 705 radio | bbox + type + `value`/`state` | **Yes** | Rendered CommonForms pages; empty / printed / handwritten fills | CC BY 4.0 | Eval set for state; born-digital |
| 6 | `nutrientdocs/form-field-v1-benchmark` | 5,914 pages, 126,938 fields | `Text`, `ChoiceButton`, `Signature` (COCO xywh) | No | Rendered; empty/filled/handwritten variants (synthetic fills) | CC BY 4.0 | Localization eval incl. handwriting clutter |
| 7 | **mini-CommonForms** ("Mind the Gaps", DAS 2026) | 47,355 imgs, 1.26M boxes, **365,253 `choice`** | `text`, `choice`, `sig` (COCO) | No | Born-digital | Apache-2.0 (HF card) | Cleaner CommonForms for localization pretraining |
| 8 | NIST SD6 (and SD2) | 5,595 (5,590) pages, 1988 IRS 1040 package, 20 form faces | Per-field text answers; checkbox "ICON" fields = 1 if marked | Yes, per named field | Binary, 12 px/mm (~300 DPI), synthesized hand-printed | NIST SRD, free download | State without boxes; would need template registration |
| 9 | Roboflow `chris-bromley-bf2yc/tax-form-checkbox-detection-ewifc` | 126 imgs | `Checked`, `Unchecked`, `Question` | Yes | Tax forms | CC BY 4.0 | Small, closest domain after #2 |
| 10 | Roboflow `docfiller/doc-form-detector` | 9,328 imgs | `text`, `signature`, `checkbox`, `list`, `radio`, `select`, `unknown` | No | Likely PDF-widget-derived | CC BY 4.0 | Same role as CommonForms |
| 11 | `bugsiesegal/form-fields-for-yolo` (HF) | 13,168 pages | 9 classes incl. `CHECKBOX`, `RADIOBUTTON` + OCR words | No | Born-digital | **None stated** | Skip unless license clarified |

Other small CC BY 4.0 Roboflow sets with checked/unchecked: `s-workspace-dte1y/government-document-omr-checkbox`
(189, Chinese checklists, handwritten ticks), `gwanjin-chun/omr-checkbox-training` (250), `srihitha-odr8e/checkbox-as1kz`
(260), `ujiboo/checkbox-detector-zxwoh` (133), `checkbox-detection-ztyeq/checkbox-kwtcz-qkbid` (843, but tight
crops of digital boxes), `ocr-9s7b7/checkbox_classify` (412, classification), `annotation-mmvj7/sed-checkboxes`
(44, Public Domain), `lucas-b-574wg/checkbox-rclnw` (603, license not shown).

**Recommendation.** Build the training set from #1 (render + synthetic marks + scan degradation, exact labels),
add #2, #3 and #9 (remapped to `checked`/`unchecked`) as real-scan data, keep our 4 samples (hand-labeled) as
the test set, and use #5 and NAF as secondary checks of the state classifier. Use #7 only if we train a
learned localizer.

---

## 1. Official blank forms (verified by download and parsing)

| Form | Official URL | Pages | AcroForm? | Checkboxes |
|---|---|---|---|---|
| 1004 (= Freddie 70), Mar 2005 | https://singlefamily.fanniemae.com/media/12371/display | 7 (612x1008 pt) | **Yes**: 429 Text, 120 RadioButton, 27 CheckBox, 2 Signature | p.1: 118 drawn squares, **118/118 covered by a widget**; p.2: 16 squares / 10 widgets; p.3: 67 squares / 14 widgets; p.6: 5 |
| 1004 (Freddie copy) | https://sf.freddiemac.com/docs/pdf/forms/70.pdf | 6 | No (`Form: none`) | Same squares: p.1 118, p.2 16, p.3 67 (53 of them 7.2 pt), p.6 5 |
| 1004MC (= Freddie 71) | https://singlefamily.fanniemae.com/media/14061/display | 2 | AcroForm flag but **0 widgets** | p.1: 51 squares (48 at 8.1 pt, 3 at 7.2 pt) |
| 1004C (= Freddie 70B), Mar 2005 | https://sf.freddiemac.com/docs/pdf/forms/70B.pdf | 7 | No | pp.1-4: 79+45, 80+26, 22+39, 12+41 squares (8.1 / 8.7 pt); p.7: 5 |
| 1004D (= Freddie 442) | https://singlefamily.fanniemae.com/media/4106/display | 2 | Yes (7 Radio, 5 CheckBox) | Not in scope, but same approach |

How it was checked: PyMuPDF `page.widgets()` for AcroForm fields and `page.get_drawings()` filtered to near-square
rects of 6-10 pt for the drawn boxes. The checkboxes are vector rectangles, not font glyphs, so every box on every
page can be extracted exactly whether or not a widget exists. The widgets add one useful feature: setting them
to "on" renders a real check glyph, which gives "checked" examples for free. Hand-drawn X/✓ marks still have to
be synthesized.

Access notes. `singlefamily.fanniemae.com` sits behind a Cloudflare JS challenge: `curl` gets 403, a real browser
works. The old `fanniemae.com/content/guide_form/1004.pdf` and `/media/document/pdf/1004` paths are 404.
`sf.freddiemac.com/docs/pdf/forms/{70,70B}.pdf` download fine with `curl`. The 1004 Desktop (media/30346) is not
fillable.

**Context to note.** UAD 3.6 / the redesigned URAR replaces the 1004 family for GSE deliveries. Per Freddie Mac's
FAQ and lender notices, legacy forms cannot be submitted to UCDP for new appraisals from **2026-11-02**. FHA/VA
have not set dates. The legacy forms remain the right target for this take-home (the samples are legacy 1004s),
but production traffic will shift to the new URAR.

## 2. Real labeled checkbox data (details)

**Roboflow `new_merge_project`** (annotation-mmvj7). 861 images, Public Domain, 5 versions, no description. The
preview shows a scanned, handwritten "Annual Recertification" housing form with YES/NO boxes, which is very close
to our domain. The class list is messy (`checked`, `unchecked`, `checkbox-3Rxo`, `checkbox-Dbj5`, `checkbox-pgxF`,
`'`, `\`), so it needs a remap and a quick audit. The same workspace has `sed-checkboxes` (44, Public Domain).

**Roboflow `checkboxes-xtv3p`** (cogniquest). 2,830 images, CC BY 4.0, classes `checked`/`unchecked`, 5 versions,
no description. The thumbnails are scanned document pages (a form, a legal page). Sources are unknown, so check
for PII and label quality before use.

**NAF dataset** (Davis et al., "Deep Visual Template-Free Form Parsing", ICDAR 2019, arXiv 1909.02576). The forms
come from the US National Archives and FamilySearch. JSON per image has `fieldBBs` with `type` (includes
`fieldCheckBox`) and `isBlank` (0 text, 1 handwriting, 2 print/stamp, 3 blank, 4 signature). A checkbox with
`isBlank != 3` should be a marked box (inferred from the schema, not verified per instance). Download
`labeled_images.tar.gz` (790 MB) from the v1.0 release. License CDLA-Permissive-1.0.

**Nutrient benchmarks** (HF, CC BY 4.0, "attribute to Nutrient"). Both come from CommonForms pages. v1 renders
real blank forms and then adds synthetic typed or handwritten values ("No PII"). The vlm-v2 benchmark carries a
per-field `value`/`state` (checked / unchecked / option), so it is the only public set with checkbox **state +
bbox** at page level that has a clear license. Its pages are rendered, not scanned. Related open model:
`nutrientdocs/form-field-v1-state`, a 96k-param ONNX filled/empty crop classifier (Apache-2.0, 0.981 acc on their
synthetic split), which is a ready baseline for our state step. The `-small`/`-medium` detectors are commercial.

**mini-CommonForms** (Brini, Moured, Gbada, Barney; arXiv 2609.23679; DAS 2026, LNCS pp. 459-473). A curated
re-annotation of CommonForms to fix missing and misplaced labels. COCO splits: train 27,338 / val 7,400 / test
12,617 images, 17.8% negatives. HF `omoured/minicommonform` (Apache-2.0 tag), GitHub `moured/mini-commonforms`.
No checked state.

**NIST SD6** (SFRS II) / **SD2** (SFRS I). Synthesized 1988 IRS forms (1040, Schedules A-F, SE, 2106, 2441, 4562,
6251) with hand-printed entries, binary images at 12 px/mm. Ground truth is per-field text, and "ICON" fields
(check marks, signatures) are recorded as 1 when present. There are **no box coordinates**, so you would register
each page to its template to get boxes. Downloads: `s3.amazonaws.com/nist-srd/SD6/sd06.zip` (970 MB),
`.../SD2/sd02.zip` (939 MB). These are the only large public sets of scanned US government forms with checkbox
state, but they are 1988 tax forms, so they are lower priority than synthesis from the real templates.

**AutoFormBench** (arXiv 2603.29832, Mar 2026). 407 real government, healthcare and enterprise forms labeled
checkbox / line / box. No public download link was found, and checked state is not mentioned.

## 3. Ruled out (no checkbox class, no boxes, or wrong domain)

| Dataset | Why not |
|---|---|
| FUNSD (199 scanned forms) / XFUND | Entity labels `header/question/answer/other`. No checkbox class. Non-commercial terms. Still useful as unlabeled scan noise. |
| SROIE | Receipts, text KIE only. |
| RVL-CDIP / IIT-CDIP | Page-level class labels (16 classes incl. `form`). No boxes. |
| DocBank | 12 token-level layout classes (abstract, author, caption, equation, figure, footer, list, paragraph, reference, section, table, title). arXiv papers. |
| PubLayNet | 5 classes (text, title, list, table, figure). PubMed papers. |
| DocLayNet | 11 classes (Caption, Footnote, Formula, List-item, Page-footer/header, Picture, Section-header, Table, Text, Title). No checkbox. CDLA-Permissive. |
| OmniDocBench (1,651 pp.) | 18 block categories, no checkbox. Research-only, no commercial use. |
| Kleister NDA / Charity | Long-document text extraction, no boxes. |
| FormNet (Google) | A model paper evaluated on FUNSD/CORD/Payment. No new dataset. No public "FormBench" was found. |
| Form-NLU (SIGIR 2023) | Australian financial forms, key-value labels. No checkbox labels found. |
| CheckboxQA | QA pairs, no boxes, CC BY-NC (already in §3 of the DL doc). |
| HF `armanbabayan/full_checkbox_dropdown_radiobutton` | 1,910 YOLO images but they are **web-page screenshots**. No card or class names. |
| HF `Rasi1610/DeathSe43_44_checkbox` | 445 scanned death certificates, but the GT is a Donut-style text JSON with no checkbox boxes. |
| HF `AlroWilde/invoice-checkmark-annotations` | ~600 invoices with pen ticks. Field-level `checked` boolean, no bbox. |
| HF `juanigsrz/checkbox-detection` | ~90 unlabeled scanned PDFs. Includes `real_appraisals.pdf`, which is **this same HomeVision challenge's pages**. |
| Most large Roboflow "checkbox" sets (UIed 8.3k, Hochschule Weserbergland 9.6k, HUAWEI 9.9k...) | Mobile/web UI screenshots. |
| Kaggle | Search API for checkbox/checkboxes/check box/checkbox detection: nothing relevant (only tiny or unlabeled uploads, and UI datasets). |
| Zenodo | Search API: no checkbox/form detection datasets. |
| Papers with Code | Shut down in 2025 (redirects to HF papers). The HF Hub search above covers it. |

## 4. Caveats

- Roboflow image counts, classes and licenses were read from each project page on 2026-10-07. They are
  user-uploaded, provenance is undocumented, and labels have not been audited. Spot-check before training.
- "Checked state derivable" for NAF is inferred from the schema description, not from inspecting annotations.
- The ranking is for *this* task. For a generic form-field localizer, mini-CommonForms and Nutrient v1 rank higher.

## Sources

- Fannie Mae 1004: https://singlefamily.fanniemae.com/media/12371/display · 1004MC: https://singlefamily.fanniemae.com/media/14061/display · 1004D: https://singlefamily.fanniemae.com/media/4106/display · 1004 Desktop: https://singlefamily.fanniemae.com/media/30346/display
- Freddie Mac Form 70: https://sf.freddiemac.com/docs/pdf/forms/70.pdf · Form 70B (=1004C): https://sf.freddiemac.com/docs/pdf/forms/70B.pdf
- UAD 3.6 / URAR redesign FAQ: https://sf.freddiemac.com/faqs/uad-and-forms-redesign · Pennymac 26-66: https://corr-east.pennymac.com/announcements/announcement-26-66
- IRS fillable PDFs (cross-check: f1040 has 73 CheckBox widgets): https://www.irs.gov/pub/irs-pdf/f1040.pdf
- Roboflow: https://universe.roboflow.com/annotation-mmvj7/new_merge_project · https://universe.roboflow.com/cogniquest-idc1w/checkboxes-xtv3p · https://universe.roboflow.com/chris-bromley-bf2yc/tax-form-checkbox-detection-ewifc · https://universe.roboflow.com/docfiller/doc-form-detector · https://universe.roboflow.com/s-workspace-dte1y/government-document-omr-checkbox · https://universe.roboflow.com/checkbox-detection-ztyeq/checkbox-kwtcz-qkbid · https://universe.roboflow.com/search?q=class%3Achecked
- NAF: https://github.com/herobd/NAF_dataset · https://arxiv.org/abs/1909.02576
- Nutrient: https://huggingface.co/datasets/nutrientdocs/form-field-v1-benchmark · https://huggingface.co/datasets/nutrientdocs/form-field-vlm-v2-benchmark · https://huggingface.co/nutrientdocs/form-field-v1-state
- mini-CommonForms: https://arxiv.org/abs/2609.23679 · https://huggingface.co/datasets/omoured/minicommonform · https://doi.org/10.1007/978-3-032-36207-0_27
- CommonForms: https://huggingface.co/datasets/jbarrow/CommonForms · https://arxiv.org/abs/2509.16506
- NIST SD6: https://www.nist.gov/srd/nist-special-database-6 · User's guide: https://s3.amazonaws.com/nist-srd/SD6/SD06_users_guide.pdf
- AutoFormBench: https://arxiv.org/abs/2603.29832 · Form-NLU: https://arxiv.org/abs/2304.01577 · CheckboxQA: https://arxiv.org/abs/2504.10419
- OmniDocBench: https://github.com/opendatalab/OmniDocBench
- HF: https://huggingface.co/datasets/bugsiesegal/form-fields-for-yolo · https://huggingface.co/datasets/juanigsrz/checkbox-detection · https://huggingface.co/datasets/armanbabayan/full_checkbox_dropdown_radiobutton · https://huggingface.co/datasets/Rasi1610/DeathSe43_44_checkbox · https://huggingface.co/datasets/AlroWilde/invoice-checkmark-annotations
