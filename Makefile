# AutoRAG-Live Development Makefile

.PHONY: help install test lint format clean docs serve-docs build

help: ## Show this help message
	@echo "AutoRAG-Live Development Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	poetry install
	poetry run pre-commit install

test: ## Run tests with coverage
	poetry run pytest tests/ --cov=autorag_live --cov-report=html --cov-report=term

test-quick: ## Run tests without coverage
	poetry run pytest tests/ -x --tb=short

lint: ## Run linting
	poetry run ruff check autorag_live/ tests/
	poetry run mypy autorag_live/
	poetry run bandit -r autorag_live/

format: ## Format code
	poetry run black autorag_live/ tests/
	poetry run isort autorag_live/ tests/

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf build/ dist/ *.egg-info/

docs: ## Build documentation
	cd docs && mkdocs build

serve-docs: ## Serve documentation locally
	cd docs && mkdocs serve

build: ## Build package
	poetry build

dev-setup: ## Complete development setup
	python scripts/dev-setup.py

config-validate: ## Validate all configuration files
	find config/ -name "*.yaml" -exec poetry run python -m autorag_live.cli.config_migrate validate {} \;

benchmark: ## Run performance benchmarks
	poetry run python -m autorag_live.cli benchmark --components all

pre-commit: ## Run pre-commit hooks on all files
	poetry run pre-commit run --all-files

update-deps: ## Update dependencies
	poetry update
	poetry run pre-commit autoupdate
