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

DATA ?= data/test
DOE ?= preds/

.PHONY: setup check demo eval

setup:
	uv sync

check:
	uv run pytest
	uv run doe-eval selfcheck
	@python3 -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	uv run doe-eval baseline --inputs build/demo_inputs/1.png build/demo_inputs/2.png build/demo_inputs/3.png build/demo_inputs/4.png --out build/check_out --iters 30
	@echo "=== ALL CHECKS PASSED ==="

demo:
	@python3 -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	uv run doe-eval baseline --inputs build/demo_inputs/1.png build/demo_inputs/2.png build/demo_inputs/3.png build/demo_inputs/4.png --out build/demo_out --iters 200

eval:
	uv run doe-eval evaluate --data $(DATA) --doe-dir $(DOE)