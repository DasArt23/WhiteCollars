DATA ?= data/validation
DOE ?= preds/val

.PHONY: setup check demo eval

setup:
	uv sync

check:
	uv run pytest
	uv run python -m doe_eval.cli selfcheck
	@uv run python -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	uv run python -m doe_eval.cli baseline --inputs build/demo_inputs/01.png build/demo_inputs/02.png build/demo_inputs/03.png build/demo_inputs/04.png --out build/check_out --iters 30
	@echo "=== ALL CHECKS PASSED ==="

demo:
	@uv run python -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	uv run python -m doe_eval.cli baseline --inputs build/demo_inputs/01.png build/demo_inputs/02.png build/demo_inputs/03.png build/demo_inputs/04.png --out build/demo_out --iters 200

eval:
	uv run python -m doe_eval.cli evaluate --data $(DATA) --doe-dir $(DOE)