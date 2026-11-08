EXAMPLE ?= src/dataset/example
MANIFEST ?= $(EXAMPLE)/manifest.json
WEIGHTS ?= weights/best-retrieval.pt
DATA ?= $(EXAMPLE)/validation
DOE ?= preds/val
DEVICE ?= cpu
IMAGE ?= whitecollars:local

.PHONY: install run app test lint format build setup check demo export-val eval docker-build docker-run

install:
	uv sync --frozen

run:
	uv run --frozen project predict --manifest "$(MANIFEST)" --split test --weights "$(WEIGHTS)" --device $(DEVICE)

app: docker-run

test:
	uv run --frozen pytest src/tests

lint:
	uv run --frozen ruff check src

format:
	uv run --frozen ruff format src

build:
	uv build

setup: install

check: test
	uv run --frozen python -m doe_eval.cli selfcheck
	@uv run --frozen python -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	uv run --frozen python -m doe_eval.cli baseline --inputs build/demo_inputs/01.png build/demo_inputs/02.png build/demo_inputs/03.png build/demo_inputs/04.png --out build/check_out --iters 30

demo:
	@uv run --frozen python -c "from doe_eval.demo import save_demo_inputs; save_demo_inputs('build/demo_inputs')"
	uv run --frozen python -m doe_eval.cli baseline --inputs build/demo_inputs/01.png build/demo_inputs/02.png build/demo_inputs/03.png build/demo_inputs/04.png --out build/demo_out --iters 200

export-val:
	uv run --frozen project export --manifest "$(MANIFEST)" --split validation --out "$(DOE)" --weights "$(WEIGHTS)" --device $(DEVICE)

eval:
	uv run --frozen python -m doe_eval.cli evaluate --data "$(DATA)" --doe-dir "$(DOE)" --device $(DEVICE)

docker-build:
	docker build -t "$(IMAGE)" .

docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 "$(IMAGE)"
