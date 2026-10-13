PY = uv run python
CONFIG ?= configs/default.yaml
.PHONY: setup test lint eda features train predict interpret report notebook all fast archive

setup:
	uv sync --python 3.13 --locked

test:
	uv run pytest --cov=copilote --cov-report=term-missing

lint:
	uv run ruff check src/copilote tests
	uv run ruff format --check src/copilote tests

eda:
	$(PY) -m copilote.run_all eda --config $(CONFIG)

features:
	$(PY) -m copilote.run_all features --config $(CONFIG)

train:
	$(PY) -m copilote.run_all train --config $(CONFIG)

interpret:
	$(PY) -m copilote.run_all interpret --config $(CONFIG)

predict:
	$(PY) -m copilote.run_all predict --config $(CONFIG)

report:
	$(PY) -m copilote.run_all report --config $(CONFIG)

notebook:
	$(PY) -m copilote.deliverables notebook --config $(CONFIG)

all: lint test
	$(PY) -m copilote.run_all all --config $(CONFIG)
	$(MAKE) notebook

fast:
	$(PY) -m copilote.run_all fast --config $(CONFIG)

archive:
	$(PY) -m copilote.deliverables archive --config $(CONFIG)
