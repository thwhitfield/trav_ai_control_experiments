.PHONY: install install-dev sync lint format typecheck test clean run help

# Default Python version
PYTHON_VERSION ?= 3.11

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

run:  ## Run the AI control paper experiment (use ARGS for additional arguments)
	uv run python scripts/ai_control_paper.py $(ARGS)

run-quick:  ## Run a quick test with minimal samples (trusted monitor defer only)
	uv run python scripts/ai_control_paper.py --limit 5 --defer-thresholds "0.2,0.5,0.8" --skip-untrusted-monitor-defer --skip-trusted-monitor-edit

# Development workflow
check: lint typecheck test  ## Run all checks (lint, typecheck, test)
