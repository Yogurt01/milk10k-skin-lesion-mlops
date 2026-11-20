# Repository tasks. Run `make help`. Recipes assume Ubuntu with uv (https://docs.astral.sh/uv/) and Python 3.12.
SHELL := /bin/bash
.DEFAULT_GOAL := help

SERVING_PY := serving/.venv/bin/python
NOTEBOOK_PY := notebooks/.venv/bin/python
PYTEST_FAST := -m "not parity and not slow"
COV_MIN ?= 90
API_BASE_URL ?= http://127.0.0.1:8000
# Jenkins (ci/compose.yaml, CI-D1..CI-D12): host folder for JENKINS_HOME and the CI model copy (same paths inside)
CI_HOME ?= $(HOME)/milk10k-ci
CI_MODEL_SRC ?= artifacts/model_package/final-P1a-kg-onnx
CI_COMPOSE = CI_HOME=$(CI_HOME) DOCKER_GID=$$(stat -c %g /var/run/docker.sock) docker compose -f ci/compose.yaml
# Local Kubernetes (k8s/README.md): minikube profile = kube context, namespace, Helm release
K8S_PROFILE ?= milk10k
K8S_NAMESPACE ?= milk10k
K8S_RELEASE ?= milk10k-api

.PHONY: help setup setup-notebooks test test-all test-report test-container smoke sample lint format hygiene serve fetch-model env export bench up down ci-secrets ci-up ci-down k8s-deploy k8s-stop tf-plan

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

test-report: ## Fast tests + JUnit XML and coverage XML in reports/ (fails below COV_MIN % coverage)
	$(SERVING_PY) -m pytest $(PYTEST_FAST) --junit-xml=reports/junit.xml \
		--cov=milk10k_serving --cov-report=xml:reports/coverage.xml --cov-report=term --cov-fail-under=$(COV_MIN)

test-container: ## HTTP parity (the 6 sample lesions) against a running API at API_BASE_URL; never the 479-lesion run
	env -u MILK10K_DATA_DIR API_BASE_URL=$(API_BASE_URL) $(SERVING_PY) -m pytest -m parity tests/test_http_parity.py \
		--deselect tests/test_http_parity.py::test_all_479_reference_lesions_through_http --junit-xml=reports/junit-parity.xml

smoke: ## Smoke test (scripts/smoke_test.py) against a running API at API_BASE_URL
	python3 scripts/smoke_test.py --base-url $(API_BASE_URL)

sample: ## The 6 sample lesions (training/sample/, gitignored) from a prepared dataset: make sample SAMPLE_DATA_DIR=<dataset>
	@test -n "$(SAMPLE_DATA_DIR)" || { echo "set SAMPLE_DATA_DIR to the prepared MILK10k dataset folder" >&2; exit 1; }
	uv run --no-project --python 3.12 --with numpy==2.5.3 --with pandas==3.0.6 --with PyYAML==6.0.3 \
		python training/sample/make_sample.py --data-dir "$(SAMPLE_DATA_DIR)"

lint: ## Ruff lint + format check
	$(SERVING_PY) -m ruff check .
	$(SERVING_PY) -m ruff format --check .

format: ## Apply ruff formatting and safe fixes
	$(SERVING_PY) -m ruff check --fix .
	$(SERVING_PY) -m ruff format .

hygiene: ## No data, images, weights or secrets in git (index + untracked files)
	python3 scripts/check_repo_hygiene.py --untracked notebooks/08_export_for_deployment.ipynb

fetch-model: ## Download the model package from the GitHub Release (~1.4 GB, resumable) and verify every sha256
	scripts/fetch_model.sh

serve: ## Run the API locally on port 8000 (needs the model package in MODEL_DIR)
	$(SERVING_PY) -m uvicorn milk10k_serving.api:app --host 0.0.0.0 --port 8000

export: ## Execute notebook 08: ONNX export, model package, parity check (needs the checkpoints + sample lesions)
	cd notebooks && .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 08_export_for_deployment.ipynb

bench: ## Serving measurements (docs/serving_measurements.md); needs MILK10K_DATA_DIR
	training/.venv/bin/python benchmarks/serving/measure.py cpu
	training/.venv/bin/python benchmarks/serving/measure.py cuda
	training/.venv/bin/python benchmarks/serving/variants.py
	$(NOTEBOOK_PY) benchmarks/serving/export.py

env: ## Create .env from .env.example with a random Grafana admin password (an existing .env is never changed)
	@if [ -f .env ]; then echo ".env exists: not changed"; else \
		python3 -c 'import re, secrets; text = open(".env.example").read(); print(re.sub(r"(?m)^GRAFANA_ADMIN_PASSWORD=.*$$", "GRAFANA_ADMIN_PASSWORD=" + secrets.token_urlsafe(24), text), end="")' > .env \
		&& chmod 600 .env && echo "created .env (random Grafana admin password; it is only in .env)"; fi

up: env fetch-model ## Compose stack: .env, fetch + verify the model, build, start, wait until every service is healthy
	docker compose up -d --build --wait

down: ## Stop the Compose stack; removes its containers and networks (keeps the Prometheus and Grafana volumes)
	docker compose down

ci-secrets: env ## Jenkins secrets in ci/secrets/ (gitignored, mode 600): random admin password once; Grafana password from .env
	python3 ci/make_secrets.py

ci-up: ci-secrets ## Jenkins on 127.0.0.1:8081: folders under CI_HOME, hard-linked model copy, build, start, wait until healthy
	@test -f ci/.env || { echo "ci/.env missing: copy ci/.env.example and set CI_DATASET_DIR" >&2; exit 1; }
	mkdir -p $(CI_HOME)/jenkins_home $(CI_HOME)/model
	@if [ ! -f $(CI_HOME)/model/final-P1a-kg-onnx/manifest.json ]; then \
		cp -al $(CI_MODEL_SRC) $(CI_HOME)/model/final-P1a-kg-onnx && \
		echo "hard-linked $(CI_MODEL_SRC) to $(CI_HOME)/model/final-P1a-kg-onnx (no extra disk)"; fi
	scripts/fetch_model.sh --check $(CI_HOME)/model/final-P1a-kg-onnx
	$(CI_COMPOSE) up -d --build --wait

ci-down: ## Stop Jenkins (JENKINS_HOME with the uv cache, and the model copy, stay under CI_HOME)
	$(CI_COMPOSE) down -v

k8s-deploy: ## Deploy the API to the running minikube profile: make k8s-deploy TAG=<image tag> [REPLICAS=n]
	@test -n "$(TAG)" || { echo "usage: make k8s-deploy TAG=<image tag> [REPLICAS=n] (a pinned tag, never latest)" >&2; exit 2; }
	minikube -p $(K8S_PROFILE) image ls | grep -qxF "docker.io/library/milk10k-api:$(TAG)" \
		|| minikube -p $(K8S_PROFILE) image load milk10k-api:$(TAG)
	kubectl --context $(K8S_PROFILE) apply -f k8s/namespace.yaml
	helm upgrade --install $(K8S_RELEASE) k8s/milk10k-api --kube-context $(K8S_PROFILE) -n $(K8S_NAMESPACE) \
		-f k8s/values-local.yaml --set image.tag=$(TAG) $(if $(REPLICAS),--set replicaCount=$(REPLICAS))
	kubectl --context $(K8S_PROFILE) -n $(K8S_NAMESPACE) rollout status deployment/$(K8S_RELEASE) --timeout=300s

k8s-stop: ## Stop the minikube profile (frees its memory; cluster, loaded images and model mount stay for the next start)
	minikube stop -p $(K8S_PROFILE)

tf-plan: ## Terraform plan for GKE (not yet implemented: P1-IAC-04)
	@echo "not yet implemented: P1-IAC-04" >&2; exit 1
