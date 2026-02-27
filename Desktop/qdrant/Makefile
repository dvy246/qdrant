.PHONY: install dev test lint format run-pipeline run-api run-ui docker-up docker-down clean

# Install the package in editable mode
install:
	pip install -e .

# Install with all optional dependencies (API + UI + dev tools)
dev:
	pip install -e ".[all]"

# Run the test suite
test:
	pytest tests/ -v

# Lint with ruff
lint:
	ruff check src/ tests/

# Auto-format with ruff
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run the end-to-end pipeline
run-pipeline:
	python -m molsearch.pipeline

# Start the FastAPI server
run-api:
	uvicorn molsearch.api_server:app --host 0.0.0.0 --port 8000 --reload

# Start the Streamlit UI
run-ui:
	streamlit run src/molsearch/streamlit_app.py

# Start Qdrant + API via Docker Compose
docker-up:
	docker compose up -d

# Stop Docker Compose services
docker-down:
	docker compose down

# Remove build artifacts and caches
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
