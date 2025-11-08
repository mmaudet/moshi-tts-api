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
