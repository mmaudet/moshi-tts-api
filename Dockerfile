# Multi-stage build for efficiency
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core API dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.4.2 \
    pydantic-settings==2.1.0 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1

# Install Moshi TTS from PyPI
RUN pip3 install --no-cache-dir moshi

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
