PY = uv run python
CONFIG ?= configs/default.yaml
SEARCH_CONFIG ?= configs/competitive.yaml
.PHONY: setup test lint eda features train predict interpret report notebook competitive advanced submission phase2 all fast archive

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

competitive:
	$(PY) -m copilote.competitive all --config $(CONFIG) --search-config $(SEARCH_CONFIG)
	$(PY) -m copilote.competitive_report --config $(CONFIG)

advanced:
	$(PY) -m copilote.advanced --config $(CONFIG)

submission:
	$(PY) -m copilote.best_submission

phase2:
	$(PY) -m copilote.phase2 audit
	$(PY) -m copilote.phase2 research
	$(PY) -m copilote.phase2 confirm
	$(PY) -m copilote.phase2 seed44
	$(PY) -m copilote.phase2_report

all: lint test
	$(PY) -m copilote.run_all all --config $(CONFIG)
	$(MAKE) notebook
	$(MAKE) competitive
	$(MAKE) advanced
	$(MAKE) submission
	$(MAKE) phase2

fast:
	$(PY) -m copilote.run_all fast --config $(CONFIG)

archive:
	$(PY) -m copilote.deliverables archive --config $(CONFIG)
