.PHONY: install install-dev sync lint format typecheck test clean run help

# Default Python version
PYTHON_VERSION ?= 3.13

# Default config file
CONFIG ?= configs/ai_control_paper.yaml

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync

install-dev:  ## Install with dev dependencies
	uv sync --all-extras

sync:  ## Sync environment with lockfile
	uv sync

lint:  ## Run linter (ruff)
	uv run ruff check .

format:  ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Run type checker (mypy)
	uv run mypy src scripts

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=src --cov=scripts --cov-report=term-missing

clean:  ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run:  ## Run with config file (use CONFIG=path/to/config.yaml)
	uv run python scripts/ai_control_paper.py $(CONFIG)

run-paper:  ## Run AI Control paper replication with default config
	uv run python scripts/ai_control_paper.py configs/ai_control_paper.yaml

# Development workflow
check: lint typecheck test  ## Run all checks (lint, typecheck, test)
