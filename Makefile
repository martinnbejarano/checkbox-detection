BACKEND = uv --directory backend

.PHONY: install models lint typecheck test eval check run

install:
	$(BACKEND) sync

# Each line of backend/models.txt is saved as backend/models/<repo name>.onnx; present files are kept.
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

test: models
	$(BACKEND) run pytest

# Markdown tables per locator x classifier combo: samples, external set, by tag, by scale, labels.
eval: models
	$(BACKEND) run python scripts/evaluate.py --scales 0.5,1,1.5 --labels

check: lint typecheck test

run:
	$(BACKEND) run uvicorn checkbox_detection.main:app --port 8000
