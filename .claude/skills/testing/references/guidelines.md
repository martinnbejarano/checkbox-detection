# Testing guidelines

What a good test looks like in this repo. The `testing` skill is the workflow for deciding *what* to test; this is
*how it should look*.

## Levels

"Write tests. Not too many. Mostly integration."

- **Integration (the bulk):** `with TestClient(create_app(Settings(...))) as client:` so lifespan runs, then
  `client.post("/detect", ...)`. Real pipeline, real model files (`make test` fetches them first), real or synthetic
  images. Choose the combo under test with `Settings(locator=..., classifier=...)`. HTTP errors, `?labels=true`,
  PDFs, ordering, `page` and the golden tests all live here.
- **Unit with real dependencies:** one stage object called directly, with its real model. Only when HTTP would make
  the case slow or opaque: the scale grid (box side × page width), colour vs grayscale input, tile edges.
- **Unit, pure:** no dependencies — `geometry.py`, `binarize.py` helpers, evaluation matching and normalization,
  label-strip geometry. Parametrize over the input classes.

Accuracy is not a unit-test concern. It is measured by `make eval` and locked by `test_golden.py`
(end-to-end accuracy ≥ a recorded floor per combo).

## Golden standard

Copy these, not whatever test happens to be nearby:

| Kind | Exemplar |
|---|---|
| Integration through HTTP, real pipeline, synthetic page | `backend/tests/test_api.py::test_detect_returns_empty_and_checked_box` |
| Integration, forced failure (the one allowed fake) | `backend/tests/test_api.py::test_unexpected_failure_is_500_without_stack_trace` |
| Pure function, parametrized over input classes | `backend/tests/test_imaging.py::test_decodes_allowed_formats_keeping_color` |

Update this table when a better exemplar lands (e.g. the first stage-level test and the first golden test).

## Where to draw the boundary

- Real: our stages, our pipeline, the app, model files, Pillow, OpenCV, pypdfium2, onnxruntime, RapidOCR.
- Fake (hand-written, in the test file): only to force a failure real code can't produce on demand, such as an
  unexpected exception for the 500 path. Assert the HTTP reaction, never which methods were called.
- Nothing external is called at runtime (no network), so there is nothing to mock. `unittest.mock` is not used.

## Anti-patterns

- **New test file when one exists for that behavior.** Extend it.
- **Change-detector tests:** asserting call order, call counts, intermediate arrays or private helpers. Assert the
  response, the returned boxes or the raised error.
- **Faking a stage to feed data.** Draw the page with `cv2` instead, or use a sample.
- **Conditional skips or assertions:** no `pytest.skip` / `skipif` for missing models (`make test` fetches them), no
  `if` inside a test body. A test that may not run protects nothing.
- **Test-only code in `src/`.** Helpers live in the test file, or in `conftest.py` when shared.
- **Sleeps or wall-clock timing asserts.** Latency is reported by `make eval`, never asserted in tests.
- **Expectations computed with the code under test** (e.g. calling `iou()` to build the expected value of an `nms`
  test). Hardcode them.

## Conventions

- **Earn the place:** each test protects observable behavior against a plausible regression no other test catches.
- **Every bugfix brings regression evidence:** a test that fails on `main` and passes with the fix, when it clears
  the bar above; otherwise say in the PR what evidence was used instead.
- **No order-dependent asserts on unordered results**, except the box order, which is part of the contract.
- **Tests never read the environment.** Build `Settings(...)` explicitly.
- **Real samples only from `samples/`**; synthetic pages are drawn in the test with `cv2.rectangle`, `cv2.line`,
  `cv2.putText`.

## Shape of a test

- Body of 25–30 lines, 45 at most. It reads as: build this page, call this, assert this outcome.
- **DAMP over DRY.** Helpers hide what is irrelevant to the case (PNG encoding, client setup, a blank page). What
  defines the case — box positions, sizes, marks, expected states — stays visible in the body.
- **Assert the whole outcome** with hardcoded values: the full list of `is_checked` states, every bbox within the
  stated tolerance, the full problem+json body. Spot-checking one field passes no matter what the rest is.
- **Discriminating fixtures:** make confusable things differ (an empty box and a checked box of different sizes and
  positions) so a swapped field or order shows up.
- **Parametrize only when the body is identical** and only the input and expected output change. Give every case an
  `id` that names the class (`ids=["empty", "x", "check", "filled"]`). Parametrized setup or an `if` in the body
  means it should be separate tests.

## Test files

```
backend/tests/
├── conftest.py          # samples_dir, models_dir, png_bytes(), blank_page()
├── test_api.py          # integration: contract, errors, labels, logging
├── test_pdf.py          # integration: multi-page PDFs through /detect
├── test_golden.py       # integration: each sample × each combo ≥ recorded floor
├── test_locate.py       # unit: scale grid, gray vs RGB, table lines, tile edges
├── test_classify.py     # unit: one case per mark class, ambiguous → unchecked
├── test_label.py        # unit: strip geometry + one real OCR read
├── test_geometry.py     # pure
├── test_imaging.py      # pure-ish: decode rules
└── test_evaluate.py     # pure: matching and label normalization
```
