# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Product Purpose

Demo page for the checkbox detection service. Drop a form (image or PDF) and it shows every checkbox the service found, whether each is checked, and the label next to it. Success means the result can be checked against the document at a glance.

## Positioning

Every detection is tied to its exact spot on the page: a row in the list and its box on the document are linked both ways, so any result can be verified by eye instead of taken on trust.

## Operating Context

- Inputs are US appraisal forms (Fannie Mae 1004, 1004MC, 1004C): dense, black-on-white printed pages with many small checkboxes.
- One endpoint, `POST /detect?labels=true`. Response: `boxes[]` with `bbox`, `is_checked`, `page`, `score`, `label`.
- Multi-page PDFs (up to 20 pages) are rendered in the browser with pdf.js; coordinates refer to the page at 300 DPI.
- Detection runs on CPU (YOLOX ≈ 250 ms/page on one M4 thread, a Railway vCPU 2–3× slower).

## Capabilities and Constraints

- States: checked, unchecked, and low confidence (`score` < 0.3, YOLOX only; OpenCV always scores 1).
- Upload limits: PNG, JPEG, TIFF, WEBP or PDF, up to 20 MB. Errors arrive as RFC 9457 problem+json (`title`, `detail`).
- `label` can be `null` when OCR reads nothing.
- Frontend and backend share only the HTTP contract. The redesign is free to change anything in `frontend/` as long as it keeps that contract.
- UI copy is in English.

## Evidence on Hand

- `samples/`: the 4 challenge pages plus rotated variants, for manual testing.
- Measured accuracy lives in `.agents/research/tuning-results.md` and the README tables from `backend/scripts/evaluate.py`. Do not show accuracy numbers that do not come from there.

## Product Principles

1. Verifiable precision: every result points to the exact place on the page that proves it.
2. Product craft: the page is evidence of judgment, every detail deliberate.
3. Speed: the result appears quickly and every interaction responds instantly.
4. The document is the hero; the interface serves the reading of it.

## Accessibility & Inclusion

Checked vs unchecked must read without hue (fill vs empty). Graphics 3:1 and text 4.5:1 on their background. Respect `prefers-reduced-motion`.
