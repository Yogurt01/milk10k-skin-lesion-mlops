# Repository tasks. Run `make help`. Recipes assume Ubuntu with uv (https://docs.astral.sh/uv/) and Python 3.12.
SHELL := /bin/bash
.DEFAULT_GOAL := help

SERVING_PY := serving/.venv/bin/python
NOTEBOOK_PY := notebooks/.venv/bin/python
PYTEST_FAST := -m "not parity and not slow"

.PHONY: help setup setup-notebooks test test-all lint format hygiene export bench up down k8s-deploy tf-plan

help: ## List the targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup: ## Serving venv (torch-free runtime + pytest/ruff) in serving/.venv
	uv venv serving/.venv --python 3.12
	uv pip install --python $(SERVING_PY) -r serving/requirements.txt -r serving/requirements-dev.txt -e serving

setup-notebooks: ## Notebook/export venv (torch, milk10k, ONNX tools) in notebooks/.venv
	uv venv notebooks/.venv --python 3.12
	uv pip install --python $(NOTEBOOK_PY) -r notebooks/requirements.txt
	uv pip install --python $(NOTEBOOK_PY) -e training -e serving

test: ## Fast tests (no model package needed)
	$(SERVING_PY) -m pytest $(PYTEST_FAST)

test-all: ## All tests, incl. parity against the reference (needs the model package and sample lesions)
	$(SERVING_PY) -m pytest

lint: ## Ruff lint + format check
	$(SERVING_PY) -m ruff check .
	$(SERVING_PY) -m ruff format --check .

format: ## Apply ruff formatting and safe fixes
	$(SERVING_PY) -m ruff check --fix .
	$(SERVING_PY) -m ruff format .

hygiene: ## No data, images, weights or secrets in git (index + untracked files)
	python3 scripts/check_repo_hygiene.py --untracked notebooks/08_export_for_deployment.ipynb

export: ## Execute notebook 08: ONNX export, model package, parity check (needs the checkpoints + sample lesions)
	cd notebooks && .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 08_export_for_deployment.ipynb

bench: ## Serving measurements (docs/serving_measurements.md); needs MILK10K_DATA_DIR
	training/.venv/bin/python benchmarks/serving/measure.py cpu
	training/.venv/bin/python benchmarks/serving/measure.py cuda
	training/.venv/bin/python benchmarks/serving/variants.py
	$(NOTEBOOK_PY) benchmarks/serving/export.py

up: ## Docker Compose stack (not yet implemented: P1-DCK-07)
	@echo "not yet implemented: P1-DCK-07" >&2; exit 1

down: ## Stop the Compose stack (not yet implemented: P1-DCK-07)
	@echo "not yet implemented: P1-DCK-07" >&2; exit 1

k8s-deploy: ## Helm deploy to minikube (not yet implemented: P1-K8S-06)
	@echo "not yet implemented: P1-K8S-06" >&2; exit 1

tf-plan: ## Terraform plan for GKE (not yet implemented: P1-IAC-04)
	@echo "not yet implemented: P1-IAC-04" >&2; exit 1
