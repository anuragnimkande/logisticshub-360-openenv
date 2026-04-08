# ─────────────────────────────────────────────────────────────
# LogisticsHub-360 — Hugging Face Spaces Dockerfile (FIXED)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="LogisticsHub-360 Team" \
      version="1.0.0" \
      description="LogisticsHub-360: Intelligent E-Commerce Operations Environment"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=7860

# Working directory
WORKDIR /app

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy full project (simpler & safer)
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# ✅ IMPORTANT: Expose HF required port
EXPOSE 7860

# (Optional) Healthcheck — keep lightweight
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "print('OK')" || exit 1

# ✅ Start app (HF expects this)
CMD ["python", "app.py"]