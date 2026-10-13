# Built from the repo root (docker build -f backend/Dockerfile .) because the image also holds the frontend.

FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# uv's recipe: dependencies first so code changes reuse that layer, then the project itself.
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# No cache mounts: Railway's builder rejects them without its own id prefix.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY backend/src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.13-slim
RUN useradd --system --no-create-home app
COPY --from=builder /app/.venv /app/.venv
COPY --from=frontend /frontend/dist /app/static
# The default pipeline runs the fine-tuned models, which are committed.
COPY backend/models/*-tuned.onnx /app/models/
ENV PATH=/app/.venv/bin:$PATH \
    CHECKBOX_STATIC_DIR=/app/static \
    CHECKBOX_MODELS_DIR=/app/models
USER app
EXPOSE 8000
HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "checkbox_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]
