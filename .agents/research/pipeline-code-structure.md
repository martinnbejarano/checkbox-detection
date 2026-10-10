# Code structure for the 3-stage pipeline: locate → classify state → label

Researched 2026-10-10. Scope: how readable open-source Python projects lay out a multi-stage ML inference pipeline
served behind an API, and what this repo should copy from them. The planned redesign has three stages:
`Locator` (OpenCV or `form-field-v1-nano` YOLOX tiled at 1280 px), `StateClassifier` (`fill_ratio` or
`form-field-v1-state` CNN) and `Labeler` (RapidOCR recognition only, only when `?labels=true`). Runtime is
onnxruntime only, and weights are fetched at Docker build time from a pinned Hugging Face revision. Already covered,
not repeated here: FastAPI layering, lifespan, threadpool and ORT threading (`api-architecture.md`), and model
choices with probe numbers (`learned-pipeline.md`).

Method: I shallow-cloned each project on 2026-10-10 (commits in Sources) and read the source files named below.
Bracketed numbers like [3] point to the Sources section. **(est.)** marks my own judgement, not a measured or
documented fact.

---

## TL;DR

- **Every project studied splits the pipeline the same way:** one object per stage that owns its model session and
  its pre/post-processing, plus one thin composer that calls the stages in order. docTR/OnnxTR
  (`OCRPredictor(det_predictor, reco_predictor)`) [3][4] and RapidOCR (`TextDetector` → `TextClassifier` →
  `TextRecognizer` behind `RapidOCR.__call__`) [5] are the cleanest examples. Docling does it with a list of
  `BasePageModel` callables (`build_pipe`) [1], but it wraps that in options classes, factories and a plugin
  registry, which is far more than we need.
- **Copy the composer pattern, not the frameworks.** We need a frozen `Pipeline` dataclass with three fields and a
  `run()` method that makes three named calls. A list of stages looped generically (docling) hides the order, and
  the order is the thing a newcomer most needs to see.
- **Interfaces: `typing.Protocol` for `Locator` and `StateClassifier`** (two implementations each), and no Protocol
  for the labeler (one implementation). Docling and natural-pdf use ABCs because their bases carry shared helpers
  (`nms`, option plumbing) [1][7]. Our helpers live in `geometry.py`, so a Protocol is enough, and test fakes need no
  inheritance. OnnxTR's own tests use exactly that kind of duck-typed fake stage [4].
- **The ONNX session is created once, outside the stage, by one small function, and passed in.** OnnxTR (`Engine`)
  and RapidOCR (`OrtInferenceSession`) both build `SessionOptions` in one place [4][5]. Both also default to
  "all cores" (`-1`), which is wrong for 1 vCPU. We should pass `intra_op_num_threads=1` explicitly.
- **Weights: plain files on disk at a path from settings. No download code in the service.** OnnxTR, RapidOCR,
  surya and docling all download at first use into a user cache [1][2][4][5]. That suits a library but not a
  container. Docling's own docs recommend prefetching (`docling-tools models download`) plus `artifacts_path` for
  offline use [1]. For us that means: `hf download <repo> model.onnx --revision <full sha> --local-dir ...` runs in
  the Dockerfile and in `make models`. The app reads `CHECKBOX_MODELS_DIR` and fails at startup if a selected model
  file is missing. `huggingface_hub` never becomes a runtime dependency.
- **Tests: synthetic images for each stage, fake stages for the composer, golden tests per combo on real samples,
  skipped when weights are absent.** surya and OnnxTR build synthetic pages with PIL [2][4]. OnnxTR tests
  `OCRPredictor` wiring with `_FixedDetPredictor`/`_DummyRecoPredictor` fakes [4]. Docling keeps golden outputs
  with a regenerate flag (`DOCLING_GEN_TEST_DATA`) [1].

### Recommendation (concrete)

Keep the package name `detection/`, because the spec, the import rules and the docstrings already use it. Inside
it, use one subpackage per stage that has more than one implementation, and a single module for the labeler.

```
backend/src/checkbox_detection/
├── main.py              # lifespan: build_pipeline(...) once → app.state.pipeline (fails fast)
├── config.py            # + locator, classifier, models_dir, onnx_threads
├── api/                 # unchanged, except ?labels=true and BoxOut.label
├── imaging/             # unchanged
└── detection/
    ├── __init__.py      # Pipeline, build_pipeline(), LOCATORS, CLASSIFIERS  ← the one file to read first
    ├── types.py         # Rect, Box
    ├── geometry.py      # iou, nms, sort_reading_order, tile_windows
    ├── ink.py           # to_gray, binarize (shared by the OpenCV locator and fill_ratio)
    ├── onnx.py          # load_session(path, threads) -> InferenceSession
    ├── locate/
    │   ├── __init__.py  # Locator Protocol
    │   ├── opencv.py    # OpenCVLocator + OpenCVParams       (today's find_candidates)
    │   └── yolox.py     # YoloxLocator + YoloxParams: tile → letterbox → run → decode → global NMS
    ├── classify/
    │   ├── __init__.py  # StateClassifier Protocol
    │   ├── fill_ratio.py  # FillRatioClassifier + FillParams (today's fill_ratio)
    │   └── state_cnn.py   # StateCnnClassifier: crop → 40×160 gray letterbox → one batched run → softmax
    └── label.py         # RapidOcrLabeler: label strip per box → RapidOCR rec-only
```

Why each module exists:

| Module | Why it earns its place |
|---|---|
| `detection/__init__.py` | One place shows the order of the stages and which names the config accepts. `evaluate.py` imports `LOCATORS`/`CLASSIFIERS` to loop over combos. |
| `types.py` | `Rect` is the contract between stage 1 and stages 2/3, and `Box` is the contract with `api/`. |
| `geometry.py` | Already exists. Gains `nms` and `tile_windows` because YOLOX tiling needs them. Pure functions, easy to test. |
| `ink.py` | Two implementations (OpenCV locator and `fill_ratio`) need the same binarization. Putting it here stops one stage importing another. |
| `onnx.py` | One function, used by three stages, so the thread setting and the "file missing" error are written once. |
| `locate/`, `classify/` | Two implementations each, chosen by config. The Protocol sits in `__init__.py` next to them. |
| `label.py` | One implementation, so it gets a module but no Protocol and no subpackage. |

Delete: `detection/opencv/` (split into `locate/opencv.py`, `classify/fill_ratio.py` and `ink.py`) and the
`Detector` Protocol (`Pipeline` replaces it as the thing `api/` depends on).

**Types (`detection/types.py`):**

```python
@dataclass(frozen=True, slots=True)
class Rect:
    x1: int; y1: int; x2: int; y2: int
    score: float  # locator confidence; 1.0 for OpenCV, which has none

@dataclass(frozen=True, slots=True)
class Box:
    rect: Rect
    is_checked: bool
    checked_score: float  # P(checked) for the CNN, interior ink ratio for fill_ratio
    label: str | None = None  # only filled when labels were requested
```

**Interfaces:**

```python
class Locator(Protocol):
    def locate(self, image: np.ndarray) -> list[Rect]: ...

class StateClassifier(Protocol):
    # Batched: all rects of a page in one call, so the CNN does one session.run (OnnxTR/RapidOCR batch the same way).
    def classify(self, image: np.ndarray, rects: list[Rect]) -> list[tuple[bool, float]]: ...

class RapidOcrLabeler:  # concrete, no Protocol
    # Takes every rect so each strip can stop at the next box on the same row.
    def label(self, image: np.ndarray, rects: list[Rect]) -> list[str]: ...
```

**Composer (`detection/__init__.py`):**

```python
@dataclass(frozen=True)
class Pipeline:
    locator: Locator
    classifier: StateClassifier
    labeler: RapidOcrLabeler

    def run(self, image: np.ndarray, labels: bool = False) -> list[Box]:
        rects = self.locator.locate(image)
        states = self.classifier.classify(image, rects)
        boxes = [Box(r, c, s) for r, (c, s) in zip(rects, states, strict=True)]
        boxes = sort_reading_order(boxes)
        if labels:
            texts = self.labeler.label(image, [b.rect for b in boxes])
            boxes = [replace(b, label=t) for b, t in zip(boxes, texts, strict=True)]
        return boxes

LOCATORS: dict[str, Callable[[Path, int], Locator]] = {
    "opencv": lambda models, threads: OpenCVLocator(),
    "yolox": lambda models, threads: YoloxLocator(load_session(models / "form-field-v1-nano.onnx", threads)),
}
CLASSIFIERS: dict[str, Callable[[Path, int], StateClassifier]] = {
    "fill_ratio": lambda models, threads: FillRatioClassifier(),
    "cnn": lambda models, threads: StateCnnClassifier(load_session(models / "form-field-v1-state.onnx", threads)),
}

def build_pipeline(locator: str, classifier: str, models_dir: Path, threads: int) -> Pipeline:
    # Unknown name → ValueError listing the choices (same as today's build_detector).
    ...
```

`detection` still must not import `config`, so `main.py` passes plain values in:
`build_pipeline(s.locator, s.classifier, s.models_dir, s.onnx_threads)`.

**ONNX session (`detection/onnx.py`):** a single function. Pre- and post-processing stay in the stage module that
owns the model (letterbox, YOLOX decode and tiling in `locate/yolox.py`; crop and 40×160 letterbox in
`classify/state_cnn.py`). That follows OnnxTR's layout, where each model file carries its own pre/post code, but
without its separate `PreProcessor` and `PostProcessor` classes [4].

```python
def load_session(path: Path, threads: int) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"Model file {path} is missing; run `make models` or set CHECKBOX_MODELS_DIR")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
```

**Weights at runtime:**

- `Settings.models_dir: Path = Path("models")` (relative to `backend/`, gitignored). The Docker image sets
  `CHECKBOX_MODELS_DIR=/app/models`.
- Fetch with `uvx --from huggingface_hub hf download nutrientdocs/form-field-v1-nano model.onnx --revision
  7cbc1ecd55d59f6315a1415dddc59c3c8b115f05 --local-dir /tmp/nano`, then copy or rename the file to
  `models/form-field-v1-nano.onnx`. Do the same for `nutrientdocs/form-field-v1-state` at
  `62b1049a066d63eacf0f3542dc155b11c7284883`. These are the current `main` commits from the HF API on 2026-10-10.
  The HF docs require the **full 40-character hash** for commit pins [9].
- Put the revisions in **one place**, as `ARG`s in the Dockerfile that `make models` reuses (or a two-line
  `models.txt` read by both). Running the fetch in a separate builder stage keeps `huggingface_hub` and `uv` out of
  the runtime layer.
- Startup is fail-fast: `build_pipeline` runs in lifespan, so a missing file stops the app before `/health` turns
  green. Only the models the config selects are loaded, so `opencv + fill_ratio` runs with no weights at all.
- RapidOCR loads its models **lazily on first call**, behind double-checked locks (`_load_det_model` etc. in
  `rapidocr/main.py`) [5]. To keep fail-fast and avoid a slow first `?labels=true` request, build the labeler in
  lifespan and run it once on a small blank strip. Pass `params={"Global.use_det": False, "Global.use_cls": False,
  "EngineConfig.onnxruntime.intra_op_num_threads": threads}`. Its models ship inside the `rapidocr` wheel, so
  nothing needs downloading **(est.: verify that the bundled rec model is used offline with `model_root_dir`
  unset)**.

**Config (`config.py`):** add `locator: str = "opencv"`, `classifier: str = "fill_ratio"`,
`models_dir: Path = Path("models")` and `onnx_threads: int = 1`. Model tuning (tile size 1280, overlap, score
threshold, NMS IoU) goes in frozen `YoloxParams`/`StateCnnParams` dataclasses next to the code, the same way
`OpenCVParams` works today, and not in env.

**Tests:**

```
backend/tests/
├── conftest.py            # samples_dir, models_dir; `needs_models` skip marker when the .onnx files are absent
├── test_geometry.py       # iou, nms, tile_windows cover the page and overlap
├── test_locate.py         # opencv: synthetic pages (existing cases). yolox (needs_models): one real sample, count > floor
├── test_classify.py       # both classifiers on synthetic crops: empty / X / ✓ / filled, parametrized over CLASSIFIERS
├── test_label.py          # cv2.putText("Urban") right of a drawn box → label contains "Urban"
├── test_pipeline.py       # fake locator + fake classifier: order, zip alignment, labels only when asked, empty page
├── test_imaging.py        # unchanged
├── test_api.py            # dependency_overrides with a fake Pipeline; ?labels=true adds "label"
└── test_golden.py         # parametrized over (locator, classifier) combos, a recorded floor per combo
```

`test_pipeline.py` is the OnnxTR pattern [4]: tiny classes such as
`class FixedLocator: def locate(self, image): return [Rect(...)]` prove the wiring with no models. Golden floors
stay as asserted thresholds, as in the spec. Docling-style snapshot files with a regenerate flag [1] are overkill
for four pages.

---

## 1. Comparison of patterns

| Concern | docling [1] | surya [2] | docTR / OnnxTR [3][4] | RapidOCR [5] | natural-pdf [7] | **Ours** |
|---|---|---|---|---|---|---|
| Stage unit | `BasePageModel` ABC, `__call__(conv_res, page_batch) -> Iterable[Page]` | `BasePredictor` subclasses (`DetectionPredictor`, …) holding model + processor | `DetectionPredictor(pre_processor, model)`, `RecognitionPredictor(...)` | `TextDetector`, `TextClassifier`, `TextRecognizer` classes, each `__call__` | `CheckboxDetector` ABC (`detect`, `is_available`, shared `nms`) | Protocol + class per implementation |
| Composition | `self.build_pipe: list[Callable]`, looped in `PaginatedPipeline._apply_on_pages` | User calls predictors; recognition takes detection results | `OCRPredictor.__call__`: explicit det → crop → reco calls | `RapidOCR.__call__` → `run_ocr_steps`, `use_det/use_cls/use_rec` flags | `CheckboxManager` picks an engine by name | `Pipeline.run`: three explicit calls |
| Choosing an implementation | Options classes + factories + entry-point plugins (`models/factories`, `models/plugins`) | Checkpoint string in settings | `ocr_predictor(det_arch="fast_base", reco_arch=...)`, `ARCHS` list check | YAML config + `params` dict override (omegaconf) | Engine name → class map | Plain dicts `LOCATORS`/`CLASSIFIERS` |
| Session / runtime | Torch mostly; ONNX in some stages | Torch; detection now client/server | `Engine` wraps `InferenceSession`; `EngineConfig` builds `SessionOptions` once | `OrtInferSession`, `_init_sess_opts`, providers config | `_session_cache` per path, CPU provider, default threads | `load_session(path, threads)` |
| Thread default | n/a | n/a | intra/inter `-1` (all cores) | intra/inter `-1`; `enable_cpu_mem_arena: false` | ORT default | `1` / `1`, explicit |
| Weights | `snapshot_download(repo, revision)`; `artifacts_path` for offline; warns and downloads if missing | Own S3 mirror + `manifest.json` check; `hf://` checkpoints; `MODEL_CACHE_DIR` | URLs with hash in the filename; `ONNXTR_CACHE_DIR`; SHA-256 prefix check | Bundled in the wheel, or URL + SHA-256 into `model_root_dir` | `hf_hub_download(repo, file, revision="v2")`; **logs and returns None on failure** | Files in `models_dir`, fetched at build time, startup error if missing |
| Results type | Pydantic `Page`, `Cluster`, docling-core types | Pydantic `TextDetectionResult`, `PolygonBox` | `Document` tree; raw `np.ndarray` between stages | Dataclass outputs (`TextRecOutput`, …) | `list[dict]` | Frozen dataclasses `Rect`, `Box` |
| Tests | Golden files + `DOCLING_GEN_TEST_DATA`; `tests/fakes/` for remote services | Synthetic PIL page (`"Hello World"`), session fixtures with real models | Synthetic text images (`synthesize_text_img`), fake det/reco predictors | Real test images in `tests/test_files`, config tests | Re-implementation tests | Synthetic + fake stages + golden floors |

## 2. Patterns by repo: what to borrow, what is overkill

### Docling [1]

- **Where:** `docling/pipeline/base_pipeline.py` (`BasePipeline`, `PaginatedPipeline._apply_on_pages` loops
  `for model in self.build_pipe: page_batch = model(conv_res, page_batch)`), `docling/models/base_model.py`
  (`BasePageModel(ABC)` with an abstract `__call__`), `docling/models/stages/{layout,ocr,table_structure,...}/`
  (one folder per stage, one file per engine, e.g. `stages/ocr/rapid_ocr_model.py`),
  `docling/models/utils/hf_model_download.py` (`snapshot_download(repo_id, revision=...)` with a cache-hit log
  line), `docling/datamodel/settings.py` (`artifacts_path: Optional[Path]`).
- **Borrow:**
  - **One folder per stage, one file per engine** (`stages/ocr/rapid_ocr_model.py`, `easyocr_model.py`, …). That
    maps directly to `locate/opencv.py` and `locate/yolox.py`.
  - **`artifacts_path` as an explicit local model directory** for offline runs, and prefetching as a separate
    command (`docling-tools models download`). Our `models_dir` plus `make models` is the same idea.
  - **Pinning the revision per model** (`_model_repo_revision` class attribute passed to `download_hf_model`).
- **Overkill / avoid:**
  - Factories with `@lru_cache` plus entry-point plugin loading (`models/factories/__init__.py`,
    `load_from_plugins(allow_external_plugins=...)`), `BaseOptions` classes with `get_options_type()`, and
    accelerator and device options. All of that exists so third parties can add engines. We have two engines per
    stage, both ours.
  - **Silent download fallback.** When `artifacts_path/<repo_folder>` is missing, the table model logs "they will
    be downloaded" and fetches at runtime (`table_structure_model_granite_vision.py`). A container should fail
    instead.
  - The generic `build_pipe` list. It is flexible, but the stage order is only visible by reading the constructor.

### surya [2]

- **Where:** `surya/common/predictor.py` (`BasePredictor`: `model_loader_cls`, `batch_size`,
  `default_batch_sizes`), `surya/common/load.py` (`ModelLoader.model()`/`.processor()`), `surya/settings.py`
  (pydantic `Settings` with `DETECTOR_MODEL_CHECKPOINT = "s3://text_detection/2025_05_07"` and `MODEL_CACHE_DIR`),
  `surya/common/s3.py` (downloads into a temp dir, checks `manifest.json`, then moves into the cache),
  `surya/detection/schema.py` (Pydantic result types per stage), `tests/conftest.py` (synthetic PIL page,
  session-scoped predictor fixtures).
- **Borrow:**
  - **Checkpoint identity as a dated or pinned string in settings, next to the stage.** Our version is the HF
    revision `ARG`.
  - **A results type per stage** (`TextDetectionResult`): our `Rect` (stage 1 output) and `Box` (pipeline
    output).
  - **Session-scoped fixtures** for loaded models in tests, so ONNX sessions are built once per test run.
- **Overkill / avoid:** `ModelLoader` subclasses, the client/server split (`DetectionPredictor` now POSTs to a
  shared model server so N workers share one copy), the own S3 mirror with manifests, and torch device/dtype
  handling. We have 4 MB of models and one process.

### docTR and OnnxTR [3][4]

- **Where:** `onnxtr/models/engine.py` (`EngineConfig` builds providers and `SessionOptions` once; `Engine`
  downloads if the path is a URL, opens `InferenceSession`, and caches input/output names),
  `onnxtr/models/detection/predictor/base.py` (`DetectionPredictor(pre_processor, model)`: preprocess → run per
  batch → remove padding), `onnxtr/models/predictor/predictor.py` (`OCRPredictor.__call__`: layout → det → crop →
  reco, with explicit calls), `onnxtr/models/detection/zoo.py` (`ARCHS` list and
  `raise ValueError(f"unknown architecture '{arch}'")`), `onnxtr/utils/data.py` (`download_from_url` with
  `hash_prefix` and `ONNXTR_CACHE_DIR`), `tests/common/test_models_zoo.py` (`_FixedDetPredictor`,
  `_DummyRecoPredictor`). docTR has the same tree with torch (`doctr/models/zoo.py: ocr_predictor` composes
  `detection_predictor` + `recognition_predictor`; `preprocessor/pytorch.py`,
  `detection/core.py: DetectionPostProcessor`) [3].
- **Borrow:**
  - **The composer takes already-built stage objects** (`OCRPredictor(det_predictor, reco_predictor)`). Our
    `Pipeline(locator, classifier, labeler)` is the same, and it is what makes fake stages trivial.
  - **The composer is explicit code**, not a loop over a list. `OCRPredictor.__call__` reads top to bottom.
  - **Name → architecture lookup with a clear error** (`ARCHS` plus `ValueError`). That is our dict plus
    `build_pipeline`.
  - **Fake-stage tests** for the composer, e.g. `test_ocrpredictor_keeps_objectness_scores_aligned` checks the
    det/reco alignment with fixed boxes and dummy words. We need the same check for `zip(rects, states)`.
  - **Session options built in one place.** Our version is `load_session`.
- **Overkill / avoid:** `NestedObject` repr trees, separate `PreProcessor`/`PostProcessor` classes per model,
  `EngineConfig` provider auto-detection (CUDA/CoreML), `run_options_provider`, and auto-download from GitHub
  releases. Our pre/post code is a few functions inside the stage module.

### RapidOCR [5]

- **Where:** `python/rapidocr/inference_engine/onnxruntime/main.py` (`OrtInferSession`: accepts a ready
  `InferenceSession`, or `model_path`, or resolves and downloads by `model_root_dir` with SHA-256; then
  `_init_sess_opts`), `python/rapidocr/config.yaml` (`Global.use_det/use_cls/use_rec`,
  `EngineConfig.onnxruntime.intra_op_num_threads: -1`, `enable_cpu_mem_arena: false`), `python/rapidocr/main.py`
  (`RapidOCR(config_path, params)` merges a flat `{"Rec.lang_type": ...}` dict into the YAML; models load lazily
  under per-stage locks), `ch_ppocr_{det,cls,rec}/main.py` (one folder per stage, each a `__call__` class),
  `utils/download_file.py` (skip if the file exists and SHA-256 matches).
- **Borrow:**
  - **Let the caller pass a ready session or a path** (`cfg.session` or `cfg.model_path`). Our stage
    constructors take an `InferenceSession` built by `load_session`, which is the same seam and easy to test.
  - **Stage switches as call flags** (`engine(img, use_det=False, use_cls=False)`). It is a precedent for
    `run(image, labels=...)` as a per-request flag instead of a separate pipeline.
  - **Batch crops sorted by width** in `TextRecognizer.__call__`. This is relevant if label strips are batched
    later.
  - **Verify the file before loading it** (`_verify_model`). We check `path.is_file()`. The pinned HF commit
    already makes the content immutable, so a SHA-256 step would be redundant **(est.)**.
- **Overkill / avoid:** YAML plus omegaconf config, six inference backends (paddle, openvino, mnn, tensorrt,
  pytorch), the model resolver with language routing, and the lazy load. Lazy loading suits a library. In a
  service it hides a missing model until the first request, so we warm up in lifespan instead.

### supervision [6]

- **Where:** `src/supervision/detection/core.py` (`@dataclass class Detections`: `xyxy (n,4)`, `confidence`,
  `class_id`, `tracker_id`, `data: dict`, `metadata: dict`, many `from_*` constructors, `merge`, `empty`),
  `detection/tools/inference_slicer.py` (`InferenceSlicer(callback, slice_wh=640, overlap_wh=100,
  overlap_filter=NMS, iou_threshold=0.5)`), `detection/utils/iou_and_nms.py` (`box_iou_batch`,
  `box_non_max_suppression`).
- **Borrow:**
  - **xyxy convention plus a confidence per box.** Our `Rect(x1, y1, x2, y2, score)` matches it, and so does the
    API's `bbox`.
  - **Slicing is a separate function from the detector:** tile → callback per tile → offset → global NMS. Write
    it as `tile_windows()` + `nms()` in `geometry.py` and use them from `YoloxLocator`.
- **Overkill / avoid:** the dependency itself (matplotlib, scipy, av, requests, … per `pyproject.toml`), and the
  struct-of-arrays `Detections` with an open `data` dict. Per-box frozen dataclasses read better and type-check
  under `mypy --strict`. Pages hold fewer than about 200 boxes, so vectorizing gains nothing **(est.)**.

### natural-pdf [7]

- **Where:** `natural_pdf/checkbox/base.py` (`CheckboxDetector(ABC)` with `detect`, `is_available` and a shared
  `nms`), `onnx_engine.py` (`_resolve_model_path`: `hf_hub_download(repo_id, filename, revision=...,
  token=False)`; `_get_session` caches `InferenceSession` per path), `checkbox_options.py`
  (`model_repo="wendys-llc/checkbox-detector"`, `model_file="checkbox_yolo12n.onnx"`, `model_revision="v2"`),
  `classifier.py` (pixel-metric classifier, separate from the detector).
- **Borrow:** the **detector and classifier are separate units** for checkboxes specifically (the same split as
  ours), and `(repo, file, revision)` held together as data next to the engine.
- **Avoid:**
  - `_resolve_model_path` **logs the error and returns `None`** when the download fails. Failures are silent, and
    a server would answer with no boxes.
  - Pinning to a tag (`"v2"`). Tags can move; a full commit hash cannot [9].
  - The ABC with option-dataclass coercion in every subclass.

### Ultralytics (structure only) [8]

- **Where:** `ultralytics/engine/predictor.py` (`BasePredictor`: `preprocess`, `inference`, `postprocess`,
  `stream_inference`, callbacks) and `ultralytics/nn/autobackend.py` (`AutoBackend` hides about 15 runtimes).
- **Borrow:** naming the three steps inside a learned stage `preprocess → run → postprocess`, as private
  functions in `locate/yolox.py`, makes the YOLOX code scannable.
- **Avoid:** everything else (sources, streaming, callbacks, AutoBackend). The license is also AGPL-3.0, so copy
  ideas only, never code.

## 3. Standards

- **src layout (PyPA) [10].** Already in use (`backend/src/checkbox_detection`). It prevents tests from importing
  the working tree by accident. Keep `training/` and `scripts/` outside `src/`, as the spec says.
- **`typing.Protocol` vs ABC [11][12].** Protocols are checked structurally: a class matches without subclassing,
  and mypy checks it at the assignment `LOCATORS: dict[str, Callable[..., Locator]]`. Use an ABC only when the
  base must share code or enforce instantiation errors. Docling and natural-pdf need that for shared helpers and
  option plumbing; we do not. Do not add `@runtime_checkable`: nothing does `isinstance` on a stage.
- **Pinned weights [9].** Use `hf_hub_download(..., revision=<sha>)` or `hf download ... --revision <sha>`. A
  commit pin must be the full-length hash. `local_dir` gives a plain file tree, plus a `.cache/huggingface/`
  metadata folder that can be deleted [9]. `HF_HUB_OFFLINE=1` makes `hf_hub_download` read the cache only and skip
  the HEAD request [13]. We only need that if we choose runtime `hf_hub_download` over plain paths; the plain path
  avoids the question.
- **ONNX Runtime session lifecycle.** Create once, share across threads, set intra-op threads explicitly. See
  `api-architecture.md` [S10][S13] and `learned-pipeline.md` §7, not repeated here. The new point from this
  research: OnnxTR and RapidOCR both default to `-1` (one thread per core) [4][5], so the explicit `1` must also
  reach RapidOCR through `EngineConfig.onnxruntime.intra_op_num_threads`.

## 4. What not to build (from what these repos carry)

Plugin or entry-point registries, factory classes, options or `EngineConfig` classes, provider auto-selection
(CUDA/CoreML), a runtime model downloader or cache manager, SHA-256 checks on top of a commit pin, lazy model
loading, a generic list-of-stages runner, struct-of-arrays detections, YAML config, a separate model server
process, and pre/post-processor classes. Each one solves a problem that libraries with many users and engines have
and that we do not.

---

## Sources

Repos were shallow-cloned on 2026-10-10. The paths above are relative to these commits.

1. Docling (MIT), commit `a8c01dc6ac269d0232214735819d529cb666e0dd`: https://github.com/docling-project/docling.
   Files: `docling/pipeline/base_pipeline.py`, `docling/models/base_model.py`, `docling/models/stages/`,
   `docling/models/factories/__init__.py`, `docling/models/utils/hf_model_download.py`,
   `docling/datamodel/settings.py`, `docling/models/stages/table_structure/table_structure_model_granite_vision.py`,
   `tests/conftest.py`, `tests/verify_utils.py`. Offline/prefetch docs:
   https://github.com/docling-project/docling/blob/main/docs/usage/advanced_options.md
2. surya (Apache-2.0 code at this commit), commit `a2363d3311773a2b6145a40458043211c50c52f4`: https://github.com/datalab-to/surya.
   Files: `surya/common/predictor.py`, `surya/common/load.py`, `surya/common/s3.py`, `surya/settings.py`,
   `surya/detection/__init__.py`, `surya/detection/schema.py`, `tests/conftest.py`, `tests/test_detection.py`
3. docTR (Apache-2.0), commit `434145cdd7dc475a6d642e11300d38c9fbc5248b`: https://github.com/mindee/doctr. Files:
   `doctr/models/zoo.py`, `doctr/models/preprocessor/pytorch.py`, `doctr/models/detection/core.py`,
   `doctr/models/detection/predictor/pytorch.py`
4. OnnxTR (Apache-2.0), commit `0ec6e88555a99f35919f14f4c9d58625f02e8cc7`: https://github.com/felixdittrich92/OnnxTR.
   Files: `onnxtr/models/engine.py`, `onnxtr/models/detection/predictor/base.py`,
   `onnxtr/models/predictor/predictor.py`, `onnxtr/models/detection/zoo.py`, `onnxtr/utils/data.py`,
   `tests/common/test_models_zoo.py`, `tests/conftest.py`
5. RapidOCR (Apache-2.0), commit `0700743fc8bfb3943cd8e40e84547f14c7d15dfc`: https://github.com/RapidAI/RapidOCR.
   Files: `python/rapidocr/main.py`, `python/rapidocr/config.yaml`,
   `python/rapidocr/inference_engine/onnxruntime/main.py`, `python/rapidocr/ch_ppocr_rec/main.py`,
   `python/rapidocr/utils/download_file.py`, `python/tests/test_det_cls_rec.py`. Docs:
   https://rapidai.github.io/RapidOCRDocs/main/
6. supervision (MIT), commit `0802a0e79ae097bfaad45a65f94cba9a2ae2a1c2`: https://github.com/roboflow/supervision.
   Files: `src/supervision/detection/core.py`, `src/supervision/detection/tools/inference_slicer.py`,
   `src/supervision/detection/utils/iou_and_nms.py`, `pyproject.toml`
7. natural-pdf (MIT), commit `99593235bb2f75467ae5bb90e6e478f00840caa7`: https://github.com/jsoma/natural-pdf.
   Files: `natural_pdf/checkbox/base.py`, `onnx_engine.py`, `checkbox_options.py`, `classifier.py`,
   `default_detector.py`
8. Ultralytics (AGPL-3.0), commit `69c95d1add13a232382e53b3aa5094ec0c7738db`:
   https://github.com/ultralytics/ultralytics. Files: `ultralytics/engine/predictor.py`,
   `ultralytics/nn/autobackend.py`
9. huggingface_hub, "Download files from the Hub" (revision pinning, full-length commit hash, `local_dir`,
   `hf download`): https://huggingface.co/docs/huggingface_hub/guides/download
10. PyPA, src layout vs flat layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
11. Python typing spec, Protocols: https://typing.python.org/en/latest/spec/protocol.html
12. mypy, Protocols and structural subtyping: https://mypy.readthedocs.io/en/stable/protocols.html
13. huggingface_hub environment variables (`HF_HUB_OFFLINE`, `HF_HOME`, `HF_HUB_CACHE`):
    https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
14. HF model API (current `sha` and file list), queried 2026-10-10:
    https://huggingface.co/api/models/nutrientdocs/form-field-v1-nano ,
    https://huggingface.co/api/models/nutrientdocs/form-field-v1-state
