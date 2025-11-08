# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Moshi TTS API is a REST API wrapper around Kyutai Labs' Moshi text-to-speech model. It provides a FastAPI-based service with bilingual support (French and English), Swagger documentation, and Docker deployment. The API generates high-quality 24kHz audio in WAV or RAW format.

## Development Commands

### Local Development
```bash
# Install dependencies (without Moshi - dummy mode)
pip install fastapi uvicorn pydantic numpy scipy python-multipart aiofiles

# Install with Moshi TTS (requires PyTorch and moshi package)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
pip install moshi

# Run the API server locally
python app_final.py
# OR
uvicorn app_final:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Testing
```bash
# Run pytest tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=./ --cov-report=xml

# Lint with flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Test the API with provided script (requires running server)
chmod +x test_api.sh
./test_api.sh
```

### Docker Development
```bash
# Docker Compose with GPU (recommended)
docker compose up -d --build

# Docker Compose without GPU (CPU-only, uses dummy model)
docker build -f Dockerfile.cpu -t moshi-tts-api:latest .
docker run -d --name moshi-tts-api -p 8000:8000 -v $(pwd)/models:/app/models moshi-tts-api:latest

# Manual Docker commands with GPU
docker build -t moshi-tts-api:latest .
docker run -d --name moshi-tts-api -p 8000:8000 -v $(pwd)/models:/app/models --gpus all moshi-tts-api:latest

# View logs
docker compose logs -f
# OR
docker logs -f moshi-tts-api

# Stop and restart
docker compose down
docker compose up -d

# Rebuild after code changes
docker compose up -d --build
```

## Architecture

### Application Files

The repository contains multiple Python files representing different stages of development:
- **app.py**: Original basic implementation using Moshi loaders directly
- **app_improved.py**: Enhanced version with better error handling
- **app_final.py**: Production-ready version with full Swagger documentation, proper validation, and comprehensive endpoints - **this is the version deployed in Docker**

The Dockerfile explicitly uses `app_final.py` as the main application (copied to `app.py` in the container).

### Core Architecture

**FastAPI Application Structure** (app_final.py):
- **Request/Response Models**: Pydantic models (TTSRequest, HealthResponse, ErrorResponse) with validation
- **Enums**: LanguageCode (fr/en), AudioFormat (wav/raw)
- **Global State**: Model loaded at startup, ThreadPoolExecutor for async synthesis
- **Audio Processing**: 24kHz mono audio, NumPy arrays converted to WAV/RAW formats
- **API Versioning**: All endpoints prefixed with `/api/v1/`

**Key Endpoints**:
- `GET /`: API info and available endpoints
- `GET /api/v1/health`: Health check with model status and device info
- `GET /api/v1/languages`: List supported languages
- `POST /api/v1/synthesize`: Main TTS endpoint (text → audio)
- `POST /api/v1/synthesize/file`: TTS from uploaded text file

**Middleware**: CORS enabled for all origins (suitable for development/internal use)

### Model Integration

The application expects to integrate with Moshi via:
1. Cloning the Moshi repository to `/tmp/moshi` (in Docker or at startup)
2. Installing Moshi with `pip install -e .`
3. Importing from `moshi.models` or `moshi` module
4. Creating a model instance with device selection (CUDA/CPU)

**Note**: The current implementation includes a dummy audio generator fallback for testing when the actual Moshi model is unavailable. This generates sine wave audio based on text length.

### Docker Architecture

Multi-stage Docker build based on `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`:
- Installs Python 3.10, system dependencies (ffmpeg, libsndfile1)
- PyTorch 2.1.0 with CUDA 12.1 support
- Clones and installs Moshi from GitHub
- Runs as non-root user (appuser) for security
- Health check on `/api/v1/health` endpoint
- Model cache persisted via volume mount at `/app/models`

## Important Implementation Details

### Audio Processing
- Sample rate: 24kHz (fixed)
- Format: Mono channel, 16-bit signed integer PCM
- WAV format: Standard WAVE file with proper headers
- RAW format: PCM data only (requires ffmpeg to convert: `ffmpeg -f s16le -ar 24000 -ac 1 -i input.raw output.wav`)

### Input Validation
- Text length: 1-5000 characters
- Whitespace is normalized automatically
- Languages: "fr" (French) or "en" (English)
- File upload: Must be UTF-8 encoded text

### Threading Model
- ThreadPoolExecutor with 2 workers for CPU-bound synthesis
- Async synthesis using `asyncio.run_in_executor()`
- Prevents blocking the FastAPI event loop during model inference

### Error Handling
- Custom exception handlers for HTTPException and ValueError
- Model availability checks before synthesis
- Graceful degradation with dummy model fallback (for development/testing)

## Environment Variables

Key environment variables that can be configured:
- `CUDA_VISIBLE_DEVICES`: Specify which GPU to use
- `HF_HOME`: Custom path for Hugging Face model cache
- `TRANSFORMERS_OFFLINE`: Disable online model downloads

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) includes:
1. **Test Job**: Python setup, dependency installation, flake8 linting, pytest with coverage
2. **Build Job**: Docker image build with caching, optional push to Docker Hub on releases
3. **Security Scan Job**: Trivy vulnerability scanning with SARIF upload

Triggers on: push to main/develop, pull requests to main, and release creation.

## Client Integration

The `client.py` file provides a Python client class (`MoshiTTSClient`) for programmatic API access with methods for synthesis, health checks, and language listing.

## Testing Strategy

The test suite (`tests/test_basic.py`) validates:
- Module imports
- API structure (enums, models)
- Client instantiation

The `test_api.sh` script provides comprehensive integration testing:
- All API endpoints
- Both languages (French/English)
- Different output formats (WAV/RAW)
- Long text handling
- Error cases
- Documentation availability

---

# Session 2025-11-08: CPU Docker Image Fix

## Problem Discovered

The CPU Docker image (`mmaudet/moshi-tts-api:cpu`) was installing a **fake placeholder package** (`moshi 0.0.0`) instead of the real Moshi package from Kyutai (`moshi 0.2.11`).

### Symptoms
- Container logged: `⚠️ Moshi library not available: No module named 'moshi.models'`
- API generated continuous sound instead of voice synthesis
- Using dummy model fallback instead of real TTS

## Root Cause Analysis

Investigated and found:
1. PyPI has **multiple versions** of the `moshi` package:
   - `0.0.0` - Empty placeholder with just `"""More to come"""`
   - `0.2.11` - Real package from Kyutai Labs with full TTS functionality

2. The installation command was correct: `uv pip install moshi --extra-index-url https://download.pytorch.org/whl/cpu`

3. The issue was **NOT** with the Dockerfile itself - testing showed it correctly installs `moshi 0.2.11`

4. The Docker Hub image was built from an **older commit** before the fixes

## Solution

### Fixed Issues (Already Committed)
1. ✅ Changed `--index-url` to `--extra-index-url` to keep PyPI access
2. ✅ Added quotes around package versions to prevent shell redirection
3. ✅ GitHub Actions successfully rebuilt both GPU and CPU images

### Latest Status
- **Docker Hub image**: `mmaudet/moshi-tts-api:cpu` - ✅ FIXED (now has moshi 0.2.11)
- **Build status**: GitHub Actions completed successfully
- **Verification**: Tested and confirmed working with real Moshi package

## For Users: How to Update

If you're experiencing the dummy audio issue:

### Option 1: Pull Latest Image (Recommended)
```bash
# Stop and remove old container
docker stop moshi-tts-api 2>/dev/null
docker rm moshi-tts-api 2>/dev/null

# Pull latest fixed image
docker pull mmaudet/moshi-tts-api:cpu

# Start new container
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v moshi-models:/app/models \
    mmaudet/moshi-tts-api:cpu

# Check logs to verify real model loaded
docker logs -f moshi-tts-api
```

### Option 2: Use Helper Script
A script has been created at `/tmp/recreate_container.sh` to automate the update process.

```bash
chmod +x /tmp/recreate_container.sh
/tmp/recreate_container.sh
```

### Verify the Fix
After updating, check that the real Moshi model is loaded:

```bash
docker exec moshi-tts-api python3 -c "import moshi; print(f'Moshi version: {moshi.__version__}')"
```

Expected output: `Moshi version: 0.2.11`

If you see `0.0.0`, you're still using the old image - try:
```bash
docker pull --no-cache mmaudet/moshi-tts-api:cpu
```

## Technical Details

### Moshi Package Versions on PyPI
```
Available versions: 0.2.11, 0.2.10, 0.2.9, ..., 0.2.1, 0.1.0, 0.0.0
Latest: 0.2.11 (real package from Kyutai Labs)
```

### Package Dependencies
The real `moshi 0.2.11` package includes:
- `torch`, `torchaudio` (can use CPU or CUDA versions)
- `aiohttp`, `huggingface-hub`, `safetensors`
- `sentencepiece`, `sounddevice`, `sphn`
- `einops`, `bitsandbytes`

### Build Verification Commands
```bash
# Check Dockerfile.cpu builds correctly
docker build -f Dockerfile.cpu -t test-cpu .

# Verify moshi version
docker run --rm test-cpu python3 -c "import moshi; print(moshi.__version__)"

# Check it has models module
docker run --rm test-cpu python3 -c "import moshi.models; print('OK')"
```

## Related GitHub Actions Run
- Workflow: "Build and Push Docker Image"
- Run ID: 19191391866
- Status: ✅ Success
- Duration: 1m28s
- Commit: "Fix shell redirection issue in Dockerfile.cpu with quoted package versions"
