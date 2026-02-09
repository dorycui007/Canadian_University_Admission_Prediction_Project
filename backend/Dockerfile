# ── Backend Dockerfile ────────────────────────────────────────────────
# Python 3.13 / FastAPI / Uvicorn
# ──────────────────────────────────────────────────────────────────────

FROM python:3.13-slim AS base

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer) ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ────────────────────────────────────────────────
COPY . .

EXPOSE 8000

# ── Default command ─────────────────────────────────────────────────
CMD ["uvicorn", "src.api.predictor:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
