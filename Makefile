.PHONY: help install test lint format type-check security clean build

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov flake8 black isort mypy bandit safety build twine

test: ## Run tests with coverage
	pytest --cov=realtime_anomaly_project --cov-report=term-missing

lint: ## Run linting checks
	flake8 realtime_anomaly_project tests
	black --check realtime_anomaly_project tests
	isort --check-only realtime_anomaly_project tests

format: ## Format code
	black realtime_anomaly_project tests
	isort realtime_anomaly_project tests

type-check: ## Run type checking
	mypy realtime_anomaly_project

security: ## Run security checks
	bandit -r realtime_anomaly_project
	safety check

clean: ## Clean up build artifacts and cache
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

check: lint type-check security test ## Run all checks

all: install check build ## Install, check, and build
