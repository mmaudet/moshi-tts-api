# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Moshi TTS API is a REST API wrapper around Kyutai Labs' Moshi text-to-speech model. It provides a FastAPI-based service with bilingual support (French and English), 44 voice presets, Swagger documentation, and flexible deployment options (Docker with GPU/CPU, or native macOS with MLX).

## Development Commands

### Local Development (Non-Docker)

```bash
# Install dependencies (without Moshi - for testing API structure)
pip install fastapi uvicorn pydantic pydantic-settings numpy scipy python-multipart aiofiles

# Install with Moshi TTS (requires PyTorch)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # For CUDA 12.6
pip install moshi

# Run the API server locally
python app.py
# OR with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Native macOS Installation (Apple Silicon)

For Mac M1/M2/M3/M4/M5 users, use MLX for optimal Metal GPU acceleration:

**Requirements:**
- macOS with Apple Silicon (ARM64)
- Python 3.10, 3.11, or 3.12 (MLX does not support Python 3.13+ yet)

```bash
# Check Python version
python3 --version  # Must be 3.10.x, 3.11.x, or 3.12.x

# If you have Python 3.13+, install a compatible version:
brew install pyenv
pyenv install 3.12
pyenv local 3.12

# Run installation script
./install-macos-mlx.sh

# Activate environment and start server
source venv-moshi-mlx/bin/activate
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Why MLX for macOS:**
- Direct Metal GPU access (Docker cannot access Metal framework)
- 2-5x faster than CPU/Docker versions
- Optimized for Apple Silicon

**Python Version Issues:**
If installation fails with "no matching distributions available for mlx", you're likely using Python 3.13+. The installation script will now detect this and provide instructions.

### Testing

```bash
# Run pytest tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=./ --cov-report=xml

# Test the API (requires running server)
chmod +x test_api.sh
./test_api.sh
```

### Docker Development

```bash
# Quick build and run (GPU)
./build-and-run.sh

# Docker Compose with GPU (recommended)
docker compose up -d --build

# Manual Docker with GPU
docker build -t moshi-tts-api:latest .
docker run -d --name moshi-tts-api -p 8000:8000 -v $(pwd)/models:/app/models --gpus all moshi-tts-api:latest

# View logs
docker compose logs -f
# OR
docker logs -f moshi-tts-api

# Rebuild after code changes
docker compose up -d --build

# Stop and remove
docker compose down
docker rm -f moshi-tts-api
```

## Architecture

### Application Structure

The codebase has a clean, modular structure:

- **app.py**: Main FastAPI application with all endpoints, model loading, and synthesis logic
- **config.py**: Type-safe configuration management using pydantic-settings
- **client.py**: Python client for programmatic API access (can be used as CLI or library)

### Key Architecture Patterns

**Configuration Management** (config.py):
- Uses pydantic-settings for type-safe configuration
- Supports `.env` file (local dev), environment variables (Docker), and defaults
- Cached singleton pattern via `@lru_cache()` for performance
- All settings documented with Field descriptions

**FastAPI Application** (app.py):
- **Pydantic Models**: TTSRequest, HealthResponse, ErrorResponse with validation
- **Enums**: LanguageCode (fr/en), AudioFormat (wav/raw), VoicePreset (44 voices)
- **Global State**: Model loaded at startup, ThreadPoolExecutor for async synthesis
- **Audio Processing**: 24kHz mono, NumPy → int16 PCM → WAV/RAW
- **API Versioning**: All endpoints prefixed with `/api/v1/`
- **CORS Middleware**: Configurable via settings

**Threading Model**:
- CPU-bound synthesis runs in ThreadPoolExecutor (2 workers default)
- Uses `asyncio.run_in_executor()` to prevent blocking FastAPI event loop
- Model lives in global state, shared across requests

**Model Integration** (app.py:262-343):
- Attempts to load real Moshi TTS model from `moshi.models.tts`
- Device selection: CUDA auto-detected or forced via `MODEL_DEVICE` env var
- Dtype: Auto (bfloat16 for CUDA, float32 for CPU) or forced via config
- CFG Distillation: Handles distilled models by setting `cfg_coef_conditioning`
- Fallback: Uses dummy sine wave generator if Moshi unavailable (for testing)

**Synthesis Flow** (app.py:370-451):
- Text → `prepare_script()` → voice selection → `make_condition_attributes()`
- Generate frames → decode with MIMI → trim to `end_steps` → convert to NumPy
- Handles both multi-speaker (voices in attributes) and single-speaker (voices as prefixes) models

### API Endpoints

All endpoints are under `/api/v1/` for versioning:

- `GET /` - API info and endpoint list
- `GET /api/v1/health` - Health check with model status, device info
- `GET /api/v1/languages` - List supported languages (fr, en)
- `GET /api/v1/voices` - List all 44 voice presets with descriptions
- `POST /api/v1/tts` - Main TTS endpoint (JSON → audio file)
- `POST /api/v1/tts/file` - TTS from uploaded text file

### Voice Presets

44 voices from 4 collections (see VoicePreset enum in app.py:127-182):
- **VCTK** (10 voices): British English speakers (p225-p234)
- **CML-TTS** (10 voices): High-quality French speakers
- **Expresso** (9 voices): English with emotions (happy, angry, calm, confused) and styles (whisper, fast, enunciated)
- **EARS** (14 voices): Diverse English speakers (subset of 50)

Voice selection: Pass `"voice": "vctk/p226_023.wav"` or use enum name `"voice": "vctk_p226"`

### Docker Architecture

**GPU Image** (Dockerfile):
- Base: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`
- Python 3.12, system deps (git, libsndfile1, build tools)
- Uses `uv` package manager (10-100x faster than pip)
- Installs PyTorch + moshi together to avoid duplicate downloads
- Runs as non-root user `appuser` (UID 1001) for security
- Health check on `/api/v1/health`
- Model cache at `/app/models` (volume mount)

**Multi-architecture**: GitHub Actions workflow supports `linux/amd64` (GPU) builds

### Configuration

Environment variables (see config.py for all options):

```bash
# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
WORKERS=1

# Model
DEFAULT_TTS_REPO=kyutai/tts-1.6b-en_fr
DEFAULT_VOICE_REPO=kyutai/tts-voices
SAMPLE_RATE=24000
MODEL_DEVICE=cuda  # or cpu, auto if not set
MODEL_DTYPE=auto   # auto, bfloat16, or float32
MODEL_N_Q=32
MODEL_TEMP=0.6
MODEL_CFG_COEF=2.0

# CORS
CORS_ORIGINS=*
CORS_CREDENTIALS=true
```

Set via `.env` file (local), Docker environment, or docker-compose.yml.

## Important Implementation Details

### Audio Processing
- Sample rate: **24kHz** (fixed, do not change without model retraining)
- Format: Mono channel, 16-bit signed integer PCM
- WAV: Standard RIFF WAVE with headers
- RAW: PCM only (convert: `ffmpeg -f s16le -ar 24000 -ac 1 -i input.raw output.wav`)

### Input Validation
- Text length: 1-5000 characters (configurable via `MAX_TEXT_LENGTH`)
- Whitespace normalized automatically (app.py:209-216)
- Languages: "fr" or "en"
- File upload: Must be UTF-8

### Error Handling
- Custom HTTPException and ValueError handlers (app.py:761-783)
- Model availability checks before synthesis
- Graceful fallback to dummy model (generates sine waves for testing)

### Startup/Shutdown
- `@app.on_event("startup")`: Loads model, handles errors gracefully
- `@app.on_event("shutdown")`: Cleans up model, empties CUDA cache, shuts down executor

## CI/CD

GitHub Actions workflow (`.github/workflows/docker-publish.yml`):
- **Triggers**: Push to main/master, PRs, tags (v*.*.*)
- **Build**: Docker image for `linux/amd64` with buildx caching
- **Push**: To Docker Hub (on non-PR events)
- **Tags**: `latest` (main branch), semver (v1.0.0 → 1.0.0, 1.0, 1), SHA
- **Description**: Updates Docker Hub README from repo README.md

Secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

## Client Integration

**Python Client** (client.py):
```python
from client import MoshiTTSClient

client = MoshiTTSClient("http://localhost:8000")
client.health_check()
client.synthesize("Hello world", language="en", output_file="output.wav")
```

**CLI**:
```bash
python client.py -t "Bonjour" -l fr -o test.wav
python client.py --health
python client.py --languages
```

## Testing Strategy

- **Unit tests**: tests/test_basic.py (module imports, API structure)
- **Integration tests**: test_api.sh (bash script testing all endpoints)
- **Pytest config**: pyproject.toml with coverage settings

## Common Tasks

### Adding a New Endpoint
1. Define Pydantic request/response models in app.py
2. Add endpoint function with `@app.get()` or `@app.post()` decorator
3. Use appropriate tags (TTS or System) for documentation
4. Add tests to test_api.sh

### Changing Model Configuration
1. Update Settings class in config.py
2. Add Field with description and default
3. Use in app.py via `settings.your_field`
4. Document in .env.example (if exists) or README

### Debugging Model Loading
Check logs for:
- "✅ Moshi TTS model loaded successfully!" - Real model loaded
- "⚠️ Using dummy model for testing" - Fallback mode (generates sine waves)
- "⚠️ PyTorch not available" - Missing PyTorch
- "⚠️ Moshi library not available" - Missing moshi package

Verify model: `docker exec moshi-tts-api python3 -c "import moshi; print(moshi.__version__)"`

### Performance Optimization
- **GPU**: Real-time or faster generation
- **CPU**: 2-10x real-time depending on CPU
- **Memory**: ~6GB for bf16 model
- **First request**: Slower (model loading and caching)
- **macOS MLX**: 2-5x faster than Docker/CPU on Apple Silicon

## Deployment Notes

- **Docker Hub**: Images at `mmaudet/moshi-tts-api:latest`
- **Model caching**: Always mount `/app/models` volume to avoid re-downloading
- **Security**: Container runs as non-root user (appuser UID 1001)
- **CORS**: Default is `*` (all origins) - restrict in production
- **Health checks**: Built into Docker with 30s interval
