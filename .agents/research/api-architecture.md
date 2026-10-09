# API architecture for the checkbox-detection service (research, 2026-10-07)

Scope: one `POST /detect` endpoint that takes a multipart image and returns
`{"boxes":[{"bbox":[x1,y1,x2,y2],"is_checked":bool}]}`. Detection is CPU-bound
(OpenCV, and maybe an ONNX model). The goal is code that is easy to read, honest
about what it doesn't do yet, and not over-built.

Versions verified on PyPI on 2026-10-07 (`https://pypi.org/pypi/<name>/json`):

| Package | Version | Notes |
|---|---|---|
| fastapi | 0.142.4 | requires Python >=3.10; dropped Pydantic v1 [S1] |
| starlette | 1.7.0 | 1.0 released 2026-03-22; `max_body_size` added in 1.6.0 [S5] |
| uvicorn | 0.54.0 | |
| litestar | 2.24.0 | |
| flask | 3.1.3 | |
| bentoml | 1.4.39 | |
| uv | 0.12.23 | |
| ruff | 0.16.10 | |
| ty | 0.0.85 | "currently in beta" [S12] |
| pyright | 1.1.414 | |
| mypy | 2.4.0 | |
| pytest | 9.1.1 | |
| onnxruntime | 1.30.0 | requires >=3.11; wheels for cp311 to cp314 |
| opencv-python-headless | 5.0.0.93 | abi3 wheel (cp37+) |
| pillow | 12.3.0 | |
| pydantic / pydantic-settings | 2.13.5 / 2.15.0 | |
| python-multipart | 0.0.32 | required by FastAPI for form/file uploads [S3] |
| httpx | 0.28.1 | used by `TestClient` |

**Python: 3.13.** It is supported until 2029-10 and every dependency above ships
wheels for it. 3.14 would also work; 3.13 is the cautious choice
(https://endoflife.date/python). Pin it in `.python-version` and
`requires-python`.

---

## 1. Framework: FastAPI

| Option | Verdict for one endpoint, one model |
|---|---|
| **FastAPI** | **Use it.** Reviewers know it. You get typed request/response models, OpenAPI at `/docs` without extra work, `lifespan` for loading the model once [S2], and `TestClient`. Plain `def` endpoints run in a threadpool automatically [S4]. |
| Litestar | Technically good (msgspec, DI). Fewer reviewers know it, and nothing here needs what it adds over FastAPI. |
| Flask | Fine, but it is WSGI. You would write upload validation and an OpenAPI schema by hand, and it gives no typed response model. |
| BentoML / Ray Serve | Built for serving many models with adaptive batching and scaling. For one model they add a framework, a runtime and their own deploy story, and reviewers would spend the hour on BentoML instead of your code. Mention them as the "next step at scale" in the writeup. |
| Triton | A GPU and multi-model inference server with no HTTP business logic. Wrong tool here. |

FastAPI 0.142 added native OpenTelemetry support [S1]. That is useful later; don't enable it now.

## 2. Project layout

Use the src layout, which stops tests from importing the working directory by
accident (https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
Draw three boundaries, and use **modules, not class hierarchies**:

1. **`api/`** handles HTTP: routing, upload validation, mapping errors to status
   codes, and the response schema. This is the only place that imports FastAPI.
2. **`detection/`** is the core: `detect(image: np.ndarray) -> list[Box]`. It is
   pure, uses only numpy and OpenCV, never imports FastAPI, and is unit-testable
   with an array. If an ML model is added, it is a constructor argument of a
   `Detector` (or a module-level function that takes the session), not a plugin system.
3. **`imaging.py`** turns bytes into a validated `np.ndarray`: Pillow decode,
   EXIF orientation, size limits. It sits between the other two and has no FastAPI import.

Rules against over-abstraction:
- No `interfaces/`, `ports/`, `adapters/`, `repositories/` or `services/` folders.
  There is no persistence and only one implementation of anything.
- No `Protocol`/ABC for "the detector" until a second implementation actually
  exists (YAGNI). A plain class or function is enough.
- Use one frozen dataclass `Box` in the core. Keep the Pydantic response model in
  `api/` and convert at the edge. That conversion is the only duplication worth paying for.
- Keep config in one `Settings` object (pydantic-settings) and pass values in
  explicitly. The core never reads env vars.

### Proposed tree

```
checkbox-detection/
├── pyproject.toml            # uv project, ruff + type checker + pytest config
├── uv.lock
├── .python-version           # 3.13
├── Makefile                  # install / lint / typecheck / test / run / docker
├── Dockerfile
├── .dockerignore
├── .pre-commit-config.yaml   # optional: ruff + ruff-format only
├── README.md                 # build/run, design, trade-offs, TODOs (the writeup)
├── src/checkbox_detection/
│   ├── __init__.py
│   ├── main.py               # create_app(): lifespan, routers, exception handlers
│   ├── config.py             # Settings(BaseSettings): limits, model path, log level
│   ├── api/
│   │   ├── routes.py         # POST /detect, GET /healthz
│   │   ├── schemas.py        # Pydantic: DetectResponse, BoxOut
│   │   └── errors.py         # domain errors -> problem+json responses
│   ├── imaging.py            # bytes -> np.ndarray (validate, decode, EXIF, limits)
│   └── detection/
│       ├── __init__.py       # exports detect / Box
│       ├── types.py          # @dataclass(frozen=True) Box
│       ├── candidates.py     # OpenCV: find square-ish contours
│       └── classify.py       # checked/unchecked (fill ratio, or ONNX model)
├── models/                   # .onnx (only if used), with a README noting its source
├── scripts/
│   └── evaluate.py           # precision/recall/F1 against labeled samples
└── tests/
    ├── conftest.py           # client fixture, sample image paths
    ├── data/                 # small sample images + expected JSON (golden)
    ├── test_detection.py     # core unit tests on synthetic numpy images
    ├── test_imaging.py       # decode/validation edge cases
    ├── test_api.py           # TestClient: 200, 413, 415, 422
    └── test_golden.py        # regression on sample images (IoU-tolerant)
```

Merge `candidates.py` and `classify.py` into one `detector.py` if both together
stay under about 200 lines. Fewer files is better until a file becomes hard to read.

## 3. Tooling

- **uv** handles the project, the lockfile, the Python version and the virtualenv.
  `uv sync --locked` in CI and Docker fails if the lockfile is stale [S6].
  Put dev tools in `[dependency-groups] dev` (PEP 735), which uv installs by default and Docker skips.
- **ruff** does both linting and formatting. It "can be used to replace Flake8
  (plus dozens of plugins), Black, isort, pydocstyle, pyupgrade, autoflake" [S7].
  Suggested `select = ["E","F","W","I","B","UP","SIM","RUF","N","PTH","S"]`.
  Leave out `ALL`, because the noise invites `noqa` clutter.
- **Type checker: mypy `--strict`** (or pyright strict). Both are stable, and
  reviewers recognise them. **ty** is fast and from Astral, but it is still
  beta (0.0.x) [S12]. Mention it as an option, but don't make the take-home's
  quality signal depend on a beta tool. OpenCV 5 ships type stubs, and numpy has had them for a long time.
- **pytest** with `testpaths=["tests"]`, `--strict-markers` and `-ra`. Leave out
  `pytest-asyncio`, because the endpoint is sync and `TestClient` is sync.
  Add `pytest-cov` only if you will show the number.
- **pre-commit** is optional. If you add it, use only the ruff hooks. The Makefile is the real entry point.
- **Makefile vs justfile:** use a Makefile. Every reviewer has `make`, while
  `just` needs installing. Keep about 6 targets: `install lint typecheck test run docker-build docker-run`.
  Optionally add `eval`.

## 4. CPU-bound inference in an async server

- **Declare the endpoint with plain `def`.** FastAPI runs `def` path operations
  "in an external threadpool that is then awaited, instead of being called
  directly (as it would block the server)" [S4]. Inside a `def` endpoint, read the
  upload synchronously with `file.file.read()` [S3]. Don't use `async def` plus
  `run_in_threadpool`: it does the same thing with more code.
- **The GIL is released in the heavy parts.** OpenCV's Python bindings wrap every
  C++ call in `ERRWRAP2`, which constructs `PyAllowThreads` (`PyEval_SaveThread`) [S9].
  ONNX Runtime does the work in C++ thread pools, and "multiple threads can invoke
  the `Run()` method on the same inference session object" [S10]. Threads
  therefore give real parallelism for the expensive parts. Only Python glue code serialises.
- **The threadpool size defaults to 40** (the AnyIO limiter) [S11]. For CPU work,
  40 concurrent inferences on a 4-core box just thrash. Keep the default, but
  bound concurrency explicitly. The simplest fix is to let the container's CPU
  limit and the worker count be the throttle. Add a `# TODO(prod):` about
  admission control, such as a semaphore with fast 503s, or a queue.
- **Oversubscription is the real risk.** ORT's default intra-op threads equal the
  number of physical cores per session [S13]. OpenCV also uses its own thread pool.
  With N uvicorn workers and T concurrent requests you get N×T×cores threads.
  Make the thread counts settings (`ORT_INTRA_OP_THREADS`, `cv2.setNumThreads`),
  defaulting to 1 or 2 when you run several workers. Leave them at the library
  default if you run a single worker per container.
- **Scale with processes, not process pools inside the app.** Run one uvicorn
  process per container and scale replicas. This is FastAPI's own recommendation
  for clustered deploys; `--workers N` is fine on a single host [S8]. A
  `ProcessPoolExecutor` inside the app adds pickling of images and an extra
  failure mode, and it gains nothing over workers. That makes it a YAGNI item.
- **Load the model once in `lifespan`.** Load it before `yield` and store it on
  `app.state` [S2]. `on_event` is deprecated, and Starlette 1.0 removed
  `on_startup`/`on_shutdown` [S2][S5]. Inject the detector into the route with a
  tiny dependency (`def get_detector(request) -> Detector: return request.app.state.detector`)
  so tests can override it with `app.dependency_overrides`.
- Warm up with one dummy inference in `lifespan`, so the first request isn't slow
  and a broken model fails at startup rather than on the first request.

## 5. Upload safety (the trust boundary, so don't skimp)

Do these checks in order, cheapest first:

1. **Body size cap.** Starlette ≥1.6 supports `max_body_size` on the app and
   routes, and `RequestBodyLimitMiddleware` (FastAPI: add the middleware, see end note). Both count real bytes, not just
   `Content-Length`, and answer **413 Content Too Large** [S5][S14]. Use it with
   about 10 MiB as a setting. Multipart file parts are *not* limited by
   `max_part_size` (that covers non-file fields only), so the body limit is the real guard [S14].
   As a fallback, check `len(data) > settings.max_upload_bytes` after reading → 413.
   UploadFile spools to disk past a threshold [S3], so memory is not the issue; time and disk are.
2. **Don't trust the `Content-Type` header or the filename.** The client sets
   both [S3]. Sniff the format instead. `Image.open(fp, formats=["PNG","JPEG","TIFF","WEBP"])`
   limits which decoders Pillow will even try [S15]. Anything else is
   **415 Unsupported Media Type**. Checking the header first is fine as a cheap pre-filter, but not as the security check.
3. **Decompression bombs.** Pillow raises `DecompressionBombError` above
   2×`MAX_IMAGE_PIXELS` and only *warns* between 1× and 2× [S15]. Set
   `Image.MAX_IMAGE_PIXELS` from settings (for example 50 MP covers a 600 dpi A4
   scan), and turn the warning into an error with `warnings.simplefilter("error", Image.DecompressionBombWarning)`
   inside the decode function, or check `img.width * img.height` against your
   own limit after `open()` (it reads only the header) and before `load()`.
   Map either case to **413** or **422** with a clear message.
4. **Corrupt or truncated files.** Catch `PIL.UnidentifiedImageError` and
   `OSError` from `open()`/`load()` → **422** (or 400). Don't call `verify()` and
   then reuse the image object, because `verify()` invalidates it. Calling `load()` is the real test.
5. **EXIF orientation.** `ImageOps.exif_transpose(img)` rotates by the
   Orientation tag and strips it [S16]. Do it **before** detection, so the returned
   bbox coordinates refer to the upright image. Write this down in the API contract
   (README), because phone photos of forms will hit this case.
6. **Normalise the mode.** Convert with `img.convert("L")` (or "RGB" if a model
   needs colour) before `np.asarray`. This handles palette, alpha, 16-bit and CMYK inputs.
7. **Multi-page TIFF.** Process the first frame only (`img.seek(0)` is the
   default) and add a `# TODO(prod): multi-page`. Alternatively, reject
   `n_frames > 1` with a 422. Either way, write it down.
8. **PDF.** Reject it with 415 and leave a TODO. Rasterising needs
   pdfium/poppler, which is a large dependency with its own attack surface.
   Out of scope for "document image".
9. **Error format.** Use RFC 9457 `application/problem+json` (`type`, `title`,
   `status`, `detail`) [S17] for every 4xx/5xx, with one exception handler per
   domain error in `api/errors.py`. Domain errors (`UnsupportedImageError`,
   `ImageTooLargeError`) are raised by `imaging.py` and contain no HTTP knowledge.
   Never leak stack traces, and log 5xx with a request id.

Return 200 with `{"boxes": []}` when nothing is detected. An empty result is not an error.

## 6. Packaging and running

**Dockerfile** (multi-stage, following the official uv guide [S6]):
- Builder stage `python:3.13-slim`, with `COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/`.
  Pin the tag, as the uv docs advise. Set `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`,
  `UV_NO_DEV=1` and `UV_PYTHON_DOWNLOADS=0`.
- Run `uv sync --locked --no-install-project` with the lockfile bind-mounted and a cache
  mount, so the dependency layer is cached. Then `COPY` the source and run
  `uv sync --locked --no-editable`.
- Runtime stage `python:3.13-slim`. Copy only `/app/.venv` (and `models/`),
  set `PATH="/app/.venv/bin:$PATH"`, and run as a non-root `USER`.
- Use **`opencv-python-headless`**, never `opencv-python`. The latter pulls in GUI/X11 libraries
  (https://github.com/opencv/opencv-python#installation-and-usage). Don't mix the two in one env.
- Use an **exec-form CMD**, so signals reach uvicorn and `lifespan` shutdown runs [S8]:
  `CMD ["uvicorn", "checkbox_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]`
  (`fastapi run` works too [S8]). Use one process per container, and set `--workers` from env only for single-host demos.
- `.dockerignore` excludes `.venv`, `.git`, `tests/data` and caches [S6].

**Health:** `GET /healthz` returns `{"status":"ok"}` once lifespan has loaded the
model. With the model loaded in lifespan, "process is up" is already close to
"ready". Add a Dockerfile `HEALTHCHECK` with python/urllib (slim images have no
curl) (https://docs.docker.com/reference/dockerfile/#healthcheck). Use one
endpoint, not separate liveness and readiness endpoints, and leave a TODO to split them for k8s.

**Config:** a single `Settings(BaseSettings)` with `env_prefix="CHECKBOX_"`,
covering `max_upload_bytes`, `max_image_pixels`, `model_path`, `log_level` and
thread counts, with sensible defaults so `make run` works with zero env vars
(https://docs.pydantic.dev/latest/concepts/pydantic_settings/). Build it once in
`create_app()`. Don't scatter `os.environ` reads.

## 7. Testing strategy

| Layer | What | How |
|---|---|---|
| Core unit | `detect()` on **synthetic** images: draw empty and filled squares with `cv2.rectangle` on a white array, then assert count, bbox within ±2 px and `is_checked`. Also test a blank page (→ `[]`) and a page with text only. | pytest, numpy. Fast and deterministic, no files needed. |
| Imaging | Valid PNG/JPEG/TIFF; EXIF-rotated JPEG (bbox follows the upright image); truncated file; non-image bytes; PDF; oversize pixels (build a tiny PNG header claiming huge dimensions, or monkeypatch `MAX_IMAGE_PIXELS` low). | Generate fixtures in-test with Pillow where possible. |
| API | `TestClient(app)` used as a context manager so lifespan runs [S18]: 200 plus schema shape, 413, 415, 422 problem+json, missing file field → 422. Override the detector with `dependency_overrides` to test HTTP behaviour in isolation. | FastAPI `TestClient` (httpx). |
| Golden/regression | 3–5 real sample images in `tests/data/` with an expected `*.json`. Match boxes by IoU ≥ 0.5 and require the same `is_checked`, rather than exact pixel equality, so harmless tweaks don't break the suite. | Run with `pytest -m golden` and keep it in the default run if fast. |
| Evaluation | `scripts/evaluate.py`: for each labeled image, match predictions to ground truth by IoU, then report detection precision/recall/F1 and checked-state accuracy on the matched boxes. Print a small table. Put the numbers in the README. | stdlib + the core package. No framework. |

Reviewers will ask how you know it works. The evaluation table plus the golden
tests answer that. Coverage percentage matters less.

## 8. Light observability

Build now:
- **stdlib `logging`** with a JSON formatter (about 15 lines, or `structlog` if
  you want it; one is enough). Configure it once in `create_app()`.
- **One small middleware.** It generates or propagates an `X-Request-ID`, times
  the request with `time.perf_counter()`, logs one line per request (method,
  path, status, duration_ms, request_id) and sets an `X-Process-Time` header
  (the pattern from https://fastapi.tiangolo.com/tutorial/middleware/).
- Inside `/detect`, log the image dimensions, `n_boxes`, `decode_ms` and
  `detect_ms`. These numbers are what you would chart first.

Leave as `# TODO(prod):` comments:
- Prometheus `/metrics` (latency histogram, in-flight requests, boxes per image).
- OpenTelemetry tracing. FastAPI 0.142 now has native support, so this is
  close to a switch-on [S1].
- Auth or API keys, rate limiting, and CORS. These usually belong at the gateway.
- Admission control (a concurrency semaphore with 503 + `Retry-After`) and request timeouts.
- Model versioning in the response (`model_version` field) and model artifact checksum verification.
- Batch endpoint, async jobs for large PDFs, and multi-page TIFF.

---

## "Do not build" list (YAGNI)

- BentoML, Ray Serve, Triton, Celery/RQ queues, or a separate worker service.
- `ProcessPoolExecutor` inside the API process. Use uvicorn workers or replicas.
- `async def` endpoint + `run_in_threadpool`. A plain `def` is equivalent.
- Abstract base classes or Protocols for the detector, repository/service/
  use-case layers, DI containers, plugin registries for "future models".
- A database, caching of results, or storage of uploaded images.
- API versioning (`/v1`), pagination, or HATEOAS for one endpoint.
- Custom config systems (YAML loaders and the like). Use pydantic-settings and env only.
- Gunicorn in front of uvicorn, or the deprecated `tiangolo/uvicorn-gunicorn-fastapi` image [S8].
- Kubernetes manifests, Helm, Terraform. A single `docker run` line is enough.
- Full metrics/tracing stacks, docker-compose with Prometheus/Grafana.
- PDF rasterisation, multi-page TIFF, URL-based image fetch (fetching URLs is an SSRF risk).
- Separate liveness and readiness endpoints, and graceful-degradation modes.
- tox/nox, multiple Python versions in CI, and coverage gates.
- A custom exception hierarchy deeper than about 3 classes.

---

## Sources

- [S1] FastAPI release notes (0.142.0 native OpenTelemetry; dropped Python 3.9 and Pydantic v1): https://fastapi.tiangolo.com/release-notes/
- [S2] FastAPI Lifespan Events (loading an ML model before `yield`; `on_event` deprecated): https://fastapi.tiangolo.com/advanced/events/
- [S3] FastAPI Request Files (python-multipart, spooled `UploadFile`, `file.file.read()` in `def`, `content_type`): https://fastapi.tiangolo.com/tutorial/request-files/
- [S4] FastAPI Concurrency and async/await (`def` endpoints run in an external threadpool): https://fastapi.tiangolo.com/async/
- [S5] Starlette release notes (1.0.0 on 2026-03-22 removed `on_startup`/`on_event`; 1.6.0 added `max_body_size`): https://starlette.dev/release-notes/
- [S6] uv Docker integration guide (pinned uv image, `UV_COMPILE_BYTECODE`, `UV_LINK_MODE`, `UV_NO_DEV`, `--locked --no-install-project`, multi-stage `--no-editable`): https://docs.astral.sh/uv/guides/integration/docker/
- [S7] Ruff docs (replaces Flake8, Black, isort and others): https://docs.astral.sh/ruff/
- [S8] FastAPI in Containers (exec-form CMD, one process per container, `--workers` for a single host, deprecated base image): https://fastapi.tiangolo.com/deployment/docker/
- [S9] OpenCV Python bindings, `ERRWRAP2` → `PyAllowThreads` (GIL released around C++ calls): https://github.com/opencv/opencv/blob/4.x/modules/python/src2/cv2_util.hpp
- [S10] ONNX Runtime high-level design ("Multiple threads can invoke the Run() method on the same inference session object"): https://onnxruntime.ai/docs/reference/high-level-design.html
- [S11] AnyIO threads (default worker thread limiter = 40): https://anyio.readthedocs.io/en/stable/threads.html
- [S12] ty README ("ty is currently in beta"): https://github.com/astral-sh/ty
- [S13] ONNX Runtime threading (intra-op default = number of physical cores; contention between sessions): https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- [S14] Starlette Requests and Middleware (`max_part_size` excludes files; `RequestBodyLimitMiddleware`/`max_body_size` → 413, counts actual bytes): https://starlette.dev/requests/ , https://starlette.dev/middleware/
- [S15] Pillow `Image` reference (`MAX_IMAGE_PIXELS`, DecompressionBombWarning/Error, the `formats=` argument to `open`): https://pillow.readthedocs.io/en/stable/reference/Image.html
- [S16] Pillow `ImageOps.exif_transpose`: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
- [S17] RFC 9457 Problem Details for HTTP APIs: https://www.rfc-editor.org/rfc/rfc9457
- [S18] FastAPI Testing Events (use `with TestClient(app)` to run lifespan): https://fastapi.tiangolo.com/advanced/testing-events/
- Package versions: PyPI JSON API, `https://pypi.org/pypi/<package>/json`, queried 2026-10-07. Python support windows: https://endoflife.date/python

Verified in source (2026-10-07): `fastapi/applications.py` on master does **not**
forward `max_body_size`, so for FastAPI use
`app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_upload_bytes)`
from `starlette.middleware.body_limit` (signature `__init__(self, app, max_body_size: int)`):
https://github.com/Kludex/starlette/blob/main/starlette/middleware/body_limit.py
