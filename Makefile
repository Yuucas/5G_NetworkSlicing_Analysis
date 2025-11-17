.PHONY: help install install-dev clean lint format test test-unit test-integration coverage docker-build docker-up docker-down train serve dashboard docs

help:
	@echo "Available commands:"
	@echo "  install          - Install production dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  clean            - Remove build artifacts and cache"
	@echo "  lint             - Run code linting (flake8, mypy, pylint)"
	@echo "  format           - Format code with black and isort"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  coverage         - Generate test coverage report"
	@echo "  docker-build     - Build Docker images"
	@echo "  docker-up        - Start all services with Docker Compose"
	@echo "  docker-down      - Stop all Docker services"
	@echo "  train            - Train RL and DL models"
	@echo "  serve            - Start API server"
	@echo "  dashboard        - Launch monitoring dashboard"
	@echo "  docs             - Build documentation"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .[dev]
	pre-commit install

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .coverage htmlcov .pytest_cache .mypy_cache
	rm -rf logs/*.log

lint:
	@echo "Running flake8..."
	flake8 src tests
	@echo "Running mypy..."
	mypy src
	@echo "Running pylint..."
	pylint src

format:
	@echo "Formatting with black..."
	black src tests
	@echo "Sorting imports with isort..."
	isort src tests

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services started. Access:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Dashboard: http://localhost:8501"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Grafana: http://localhost:3000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

train:
	python scripts/train_rl_agent.py --algorithm dqn --episodes 10000
	python scripts/train_dl_model.py --model lstm --epochs 100

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/monitoring/dashboard.py

docs:
	mkdocs build
	@echo "Documentation built in site/"

setup-data:
	mkdir -p data/raw data/processed data/interim
	cp "dataset/Quality of Service 5G.csv" data/raw/

preprocess:
	python scripts/preprocess_data.py

simulate:
	python scripts/simulate_network.py --duration 3600 --users 1000
