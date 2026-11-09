.PHONY: setup lint format test test-integration eval-smoke eval-full eval-retrieval up obs-up down

setup: ## Install deps and git hooks
	uv sync
	uv run pre-commit install

lint: ## Lint, format check and type check
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

test: ## Unit tests (no network)
	uv run pytest -m "not integration"

test-integration: ## Integration tests (run `make up` first)
	uv run pytest -m integration

eval-smoke: ## Golden tasks tagged "smoke" (real model calls; runs on every PR)
	uv run python -m goalsmith.evals.runner --tag smoke --name smoke --out evals/results/smoke.md

eval-full: ## All golden tasks (real model calls)
	uv run python -m goalsmith.evals.runner --name full --out evals/results/full.md --json evals/results/full.json

eval-retrieval: ## Retrieval metrics (hit@5, MRR, nDCG) on the labelled queries
	uv run python -m goalsmith.evals.retrieval

up: ## Start Docker Compose services and wait until they are healthy
	docker compose up -d --wait

obs-up: ## Also start Langfuse for traces (UI: http://localhost:3000)
	docker compose --profile observability up -d --wait

down: ## Stop all Docker Compose services, Langfuse included (data volumes are kept)
	docker compose --profile observability down
