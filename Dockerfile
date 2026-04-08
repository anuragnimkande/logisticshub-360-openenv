# ─────────────────────────────────────────────────────────────────────────────
# LogisticsHub-360 — Production Dockerfile
# Base: python:3.11-slim
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata labels
LABEL maintainer="LogisticsHub-360 Team" \
      version="1.0.0" \
      description="LogisticsHub-360: Intelligent E-Commerce Operations Environment"

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_BASE_URL="https://router.huggingface.co/v1" \
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
    LH360_MAX_RETRIES="3" \
    LH360_TEMPERATURE="0.2"

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY env/ ./env/
COPY inference.py .
COPY configs/ ./configs/
COPY openenv.yaml .

# Create a non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Health check — verifies environment imports correctly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from env.environment import LogisticsHub360Env; env = LogisticsHub360Env('order_tracking'); env.reset(); print('OK')" || exit 1

# Default entrypoint: run the full inference evaluation
ENTRYPOINT ["python", "inference.py"]

# Default args (run all tasks)
CMD ["--task", "all"]
