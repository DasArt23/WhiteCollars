UV ?= uv
UV_FLAGS ?=
RUN = $(UV) run --frozen $(UV_FLAGS)
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))
export OMP_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
DATA ?= src/dataset/data/test
DOE ?= preds/test

.PHONY: setup check demo eval install run test lint format build gpu-check data-gray

setup:
	$(UV) sync --frozen $(UV_FLAGS)

check:
	$(RUN) pytest
	$(RUN) python -m doe_eval.cli selfcheck
	@$(RUN) python -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	$(RUN) python -m doe_eval.cli baseline --inputs build/demo_inputs/01.png build/demo_inputs/02.png build/demo_inputs/03.png build/demo_inputs/04.png --out build/check_out --iters 30
	@echo "=== ALL CHECKS PASSED ==="

demo:
	@$(RUN) python -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	$(RUN) python -m doe_eval.cli baseline --inputs build/demo_inputs/01.png build/demo_inputs/02.png build/demo_inputs/03.png build/demo_inputs/04.png --out build/demo_out --iters 200

eval:
	$(RUN) python -m doe_eval.cli evaluate --data "$(DATA)" --doe-dir "$(DOE)"

install: setup

run:
	$(RUN) project --help

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src

format:
	$(RUN) ruff format src

build:
	$(UV) build

gpu-check:
	$(RUN) python -m model.gpu_check

data-gray:
	$(RUN) python -m dataset.expand_grayscale --out build/data_gray --extra-train 1000 --stress-count 100
