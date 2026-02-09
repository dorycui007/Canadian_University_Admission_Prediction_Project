.PHONY: local test lint format clean docker

# ── Local development ───────────────────────────────────────────────
local:
	uvicorn src.api.predictor:app --host 0.0.0.0 --port 8000 --reload

# ── Install dependencies ────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Tests ───────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src --cov-report=term-missing

# ── Code quality ────────────────────────────────────────────────────
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# ── Docker ──────────────────────────────────────────────────────────
docker:
	docker compose -f ../docker-compose.yml up --build backend mongodb weaviate

# ── Cleanup ─────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
