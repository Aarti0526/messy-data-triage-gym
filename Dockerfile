# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir ".[baseline,test]" && \
    pip install --no-cache-dir build

COPY . .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "data_triage_env.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
