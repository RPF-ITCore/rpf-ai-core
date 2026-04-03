# RPF AI Core — FastAPI backend
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal OS deps for SSL, wheels, and common native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p assets/files assets/database

EXPOSE 8000

# Match Procfile pattern; tune workers with WEB_CONCURRENCY (default 2)
ENV WEB_CONCURRENCY=2

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/healthcheck >/dev/null || exit 1

CMD ["sh", "-c", "gunicorn main:app -w ${WEB_CONCURRENCY} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000"]
