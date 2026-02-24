# ── AI-BOS Dockerfile ──────────────────────────────────────────────────────────
# Production-grade multi-stage build for Streamlit + RAG pipeline
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="AI-BOS Team"
LABEL version="2.0"
LABEL description="AI Business Operating System — RAG-powered BI Platform"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency stage ──────────────────────────────────────────────────────────
FROM base AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM deps AS runtime
WORKDIR /app

# Non-root user for security
RUN addgroup --system aibos && adduser --system --group aibos
COPY --chown=aibos:aibos . .

# Create required dirs
RUN mkdir -p /app/tests /app/.streamlit /app/demo_data && \
    chown -R aibos:aibos /app

USER aibos

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint
CMD ["streamlit", "run", "ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
