install:
	uv sync

run:
	uv run project

test:
	uv run pytest src/tests

lint:
	uv run ruff check src

format:
	uv run ruff format src

build:
	uv build