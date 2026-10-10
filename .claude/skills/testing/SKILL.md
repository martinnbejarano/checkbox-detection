---
name: testing
description: "Decide what to test and write the tests for this repo (pytest). Use when: (1) implementing any ticket — plan the tests before writing code; (2) the user asks to write, add or improve tests; (3) verifying a bugfix with a regression test; (4) reviewing whether a PR's tests are worth keeping."
---

# Writing tests

For a Python 3.13 / pytest / FastAPI service with OpenCV and ONNX stages and no database. Read `references/guidelines.md` before planning: it says what
a good test looks like here. The strategy (levels, files, when tests run) is in `.agents/spec.md` §8.

Philosophy: **"Write tests. Not too many. Mostly integration."** Tests exist to catch regressions that matter, not to
raise coverage.

## Workflow

### 1. Read the code

- Read the code under test and the existing tests. Extend a test file before creating a new one.
- Existing tests are not automatically a reference. Compare them against the golden-standard list in
  `references/guidelines.md`; if they diverge, the guidelines win.

### 2. Plan: what are we testing?

Partition the input space into **equivalence classes** — inputs the code treats the same way. A 20 px box and a
25 px box are one class; a box that touches a table line is another. One test per class.

- **Happy paths:** which kinds of input are supported?
- **Failures:** scan `references/failure-modes.md`; for each mode that applies, name the input that triggers it.

Start black-box (partition by the contract), then read the implementation to refine: it shows edge cases the
contract misses and cases the code treats identically (collapse those). Assert observable behavior only, so a
refactor that keeps behavior never breaks a test. **Litmus test: if one realistic code edit could break input A but
not input B, they are different classes.**

For each candidate test write **one sentence: the realistic edit that would make it fail.**
*"If someone drops the saturation mask in `to_clean_gray`, the red watermark is found as boxes and this fails."*
"Increases coverage", "good practice" or "the function exists" do not count. No sentence → no test.

Also check:

1. **Nothing already catches it** — otherwise extend that test or add a parametrize row.
2. **We own it** — test our logic, not that Pillow decodes PNG or onnxruntime runs a model (but DO test how our
   code reacts when they fail).

### Choose the level

Defined in `references/guidelines.md`. **Integration is the default**; a unit test needs a reason.

- **Integration:** `POST /detect` through `TestClient(create_app(...))` with the real pipeline, real model files and
  real or synthetic images.
- **Unit, real dependencies:** one stage called directly (`OpenCVLocator().locate(page)`) when going through HTTP
  would make the case slow or opaque (e.g. the 10–100 px × 1000–5000 px scale grid).
- **Unit, pure:** functions with no dependencies: `iou`, `nms`, `tile_windows`, evaluation matching, label-strip
  geometry.

Never fake a stage we own to feed data. The single exception: forcing a failure the real code cannot produce on
demand (an unexpected exception → 500). Even then, assert the observable reaction, not calls.

### Build two tables (put both in the PR description)

**Tests to write:**

| Test | Level | Why it earns its place — the edit that breaks it, and why that edit is realistic |
|---|---|---|
| Red watermark page gives no boxes | Integration | If the HSV mask in `to_clean_gray` is removed or its saturation threshold raised, watermark strokes become candidates |
| Box touching a table rule is found | Unit (locator) | If the line-removal kernel is made shorter than the box side, box edges are erased with the rule |
| `?labels=true` adds `label`; default omits it | Integration | If the route always calls the labeler or the schema drops `exclude_none`, the challenge shape changes |

**Tests to skip:**

| Skip | Why |
|---|---|
| `Settings` defaults | Locks a constant; no behavior |
| onnxruntime loads a valid file | Library behavior |
| `Pipeline.run` calls stages in order with fakes | Trivial wiring; every integration test exercises it |

### Red flags: too many tests

- Testing library behavior (Pillow, OpenCV, onnxruntime, RapidOCR, pydantic).
- Vague justifications ("critical", "error-prone") — which bug in *our* code?
- Testing one-liners, constants, `*Params` values or pass-through calls.
- Several tests for one failure mode — merge, or make them parametrize rows.
- Unit tests where an integration test would work.
- Accuracy asserted in unit tests — accuracy belongs to `make eval` and `test_golden.py`.

### Filtering mindset

**Look for reasons NOT to write each test.** It must protect observable behavior against a plausible regression
that no existing test catches. Skip tests whose only value is locking constants, configuration, wiring, library
behavior or the current implementation.

### 3. Write

- Follow `references/guidelines.md`: short DAMP bodies, hardcoded expectations, full equality on the outcome.
- **The code must match the table.** A case discovered while writing goes into the table first, with its sentence.

### 4. Run

```bash
uv --directory backend run pytest tests/test_x.py -k name   # focused first
make check                                                 # lint + typecheck + full suite before the PR
```

### 5. Verify — break the code

For at least one new test per PR, introduce the bug it is meant to catch, confirm it fails, revert. Say which in the
PR. A test that always passes is worthless.

### 6. Self-check

- Every written test is in the table; every obvious skipped case has a reason.
- No anti-pattern from `references/guidelines.md`.
- For each critical branch you can name the test that fails if the branch is deleted.
- Boundaries covered or skipped on purpose: empty page, no boxes, min/max box side, page edge, tile edge.
- Every unit test is justified over an integration test.
- `make check` passes.
