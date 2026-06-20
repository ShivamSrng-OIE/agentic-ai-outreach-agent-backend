IMAGE ?= psview-agent-api
PORT ?= 8000

.PHONY: sync format lint typecheck test check run docker-build

sync:
	uv sync --all-groups

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check:
	uv lock --check
	uv sync --frozen --all-groups
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests
	uv run pytest

run:
	uv run uvicorn psview_agent.main:app --host 0.0.0.0 --port $(PORT)

docker-build:
	docker build -t $(IMAGE) .
