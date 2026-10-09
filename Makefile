BACKEND = uv --directory backend

.PHONY: install lint typecheck test check run

install:
	$(BACKEND) sync

lint:
	$(BACKEND) run ruff format --check .
	$(BACKEND) run ruff check .

typecheck:
	$(BACKEND) run mypy

test:
	$(BACKEND) run pytest

check: lint typecheck test

run:
	$(BACKEND) run uvicorn checkbox_detection.main:app --port 8000
