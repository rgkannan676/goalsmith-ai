.PHONY: setup lint format test test-integration eval-smoke eval-full up down

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

up: ## Start Docker Compose services (added in M3)
	@if [ -f docker-compose.yml ]; then docker compose up -d; else echo "no docker-compose.yml yet (M3)"; fi

down: ## Stop Docker Compose services
	@if [ -f docker-compose.yml ]; then docker compose down; else echo "no docker-compose.yml yet (M3)"; fi
