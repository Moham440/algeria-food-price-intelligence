# ============================================================
# Algeria Food Price Intelligence System — Docker Image
# ============================================================
# Build: docker build -t algeria-food-price .
# Run:   docker run -p 8501:8501 algeria-food-price
# ============================================================

FROM python:3.10-slim AS base

# System dependencies for Prophet, TensorFlow, and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Builder stage ─────────────────────────────────────────────────────────────
FROM base AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Final stage ───────────────────────────────────────────────────────────────
FROM base AS final

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Create data directories
RUN mkdir -p data/raw/fao data/raw/wfp data/processed data/external \
    models/saved logs .cache

# ── Environment Variables ──────────────────────────────────────────────────────
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 8501   # Streamlit dashboard
EXPOSE 8000   # FastAPI (optional)

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Entry-point ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
