BACKEND = uv --directory backend
FRONTEND = npm --prefix frontend

.PHONY: install models samples lint typecheck test eval check run frontend-dev frontend-build docker-build docker-run

install:
	$(BACKEND) sync
	$(FRONTEND) ci

# Pretrained models for backend/training/. Each line of backend/models.txt is saved as
# backend/models/<repo name>.onnx; present files are kept. The service runs the committed *-tuned.onnx.
models:
	@grep -Ev '^(#|$$)' backend/models.txt | while read -r repo file sha; do \
		out=backend/models/$${repo#*/}.onnx; \
		[ -f $$out ] || { curl -fsSL --create-dirs -o $$out.part \
			https://huggingface.co/$$repo/resolve/$$sha/$$file && mv $$out.part $$out; } || exit 1; \
	done

lint:
	$(BACKEND) run ruff format --check .
	$(BACKEND) run ruff check .

typecheck:
	$(BACKEND) run mypy

# Evaluation pages and ground truth, left out of the zip distribution; a present folder is kept.
samples:
	@[ -d samples ] || curl -fsSL https://github.com/martinnbejarano/checkbox-detection/archive/main.tar.gz \
		| tar -xz --strip-components=1 checkbox-detection-main/samples checkbox-detection-main/backend/tests/data

test: samples
	$(BACKEND) run pytest

# Markdown tables per locator x classifier combo: samples, external set, by tag, by scale, labels.
eval: samples
	$(BACKEND) run python scripts/evaluate.py --scales 0.5,1,1.5 --labels

check: lint typecheck test frontend-build

run:
	$(BACKEND) run uvicorn checkbox_detection.main:app --port 8000

frontend-dev:
	$(FRONTEND) run dev

# tsc + oxlint + vite build into frontend/dist
frontend-build:
	$(FRONTEND) run lint
	$(FRONTEND) run build

docker-build:
	docker build -f backend/Dockerfile -t checkbox-detection .

docker-run:
	docker run --rm -p 8000:8000 checkbox-detection
