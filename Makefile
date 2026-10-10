BACKEND = uv --directory backend

.PHONY: install models lint typecheck test check run

install:
	$(BACKEND) sync

# Each line of backend/models.txt is saved as backend/models/<repo name>.onnx; present files are kept.
models:
	@grep -v '^#' backend/models.txt | while read -r repo file sha; do \
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

check: lint typecheck test

run:
	$(BACKEND) run uvicorn checkbox_detection.main:app --port 8000
