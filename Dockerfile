# Multi-stage build for efficiency
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    curl \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv (fast Python package installer)
# Using official installation script
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create app user for security
# Use UID 1001 to avoid conflict with ubuntu user (UID 1000) in Ubuntu 24.04
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app /app/models && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install PyTorch with CUDA 12.6 support first using uv
# uv is 10-100x faster than pip
RUN uv pip install --system --break-system-packages \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install core API dependencies using uv
# Updated versions for Python 3.12 compatibility
RUN uv pip install --system --break-system-packages \
    fastapi>=0.104.1 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.4.2 \
    pydantic-settings>=2.1.0 \
    numpy>=1.26.0 \
    scipy>=1.11.4 \
    python-multipart>=0.0.6 \
    aiofiles>=23.2.1

# Install Moshi TTS from PyPI using uv
RUN uv pip install --system --break-system-packages moshi

# Copy application files
COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser config.py /app/config.py

# Switch to non-root user for security
USER appuser

# Expose port
EXPOSE 8000

# Health check with correct endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
