.PHONY: setup lint format test test-integration eval-smoke eval-full eval-retrieval up down

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

eval-smoke: ## Quick eval suite (added in M5)
	@echo "eval-smoke: not implemented yet (M5)"

eval-full: ## Full nightly eval suite (added in M5)
	@echo "eval-full: not implemented yet (M5)"

eval-retrieval: ## Retrieval metrics (hit@5, MRR, nDCG) on the labelled queries
	uv run python -m goalsmith.evals.retrieval

up: ## Start Docker Compose services and wait until they are healthy
	docker compose up -d --wait

down: ## Stop Docker Compose services (data volumes are kept)
	docker compose down
