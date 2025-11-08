# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-08

### Added
- **MLX Backend Support** for native macOS Apple Silicon (M1/M2/M3/M4/M5)
  - Automatic backend detection (MLX → PyTorch → dummy)
  - Direct Metal GPU acceleration on macOS
  - 2-5x faster than Docker/CPU on Apple Silicon
  - Native installation script `install-macos-mlx.sh`
- Python version compatibility check (3.10-3.12 required for MLX)
- `.python-version` file for pyenv users (Python 3.12.12)
- 44 voice presets from VCTK, CML-TTS, Expresso, and EARS collections
- `/api/v1/voices` endpoint to list all available voices

### Changed
- Health endpoint now reports backend type: "mlx (Metal GPU)", "cuda (GPU)", or "cpu"
- Dual backend architecture in `app.py` with automatic detection
- Updated endpoint from `/api/v1/synthesize` to `/api/v1/tts` for consistency

### Performance
- GitHub Actions Docker builds now use cache (5-10x faster builds)
- MLX backend: ~0.5s model loading time
- MLX synthesis: 2-3x real-time speed on Apple Silicon

### Fixed
- Python version check prevents installation errors with Python 3.13+
- Clear error messages for incompatible Python versions

## [1.0.0] - 2024-01-XX

### Added
- Initial release of Moshi TTS API
- FastAPI server with Swagger documentation
- Docker support with GPU acceleration
- Bilingual support (French and English)
- WAV and RAW audio format output
- Health check endpoint
- Python client for easy integration
- Comprehensive test suite
- API documentation with Swagger UI and ReDoc
- Non-root user in Docker for security
- CORS support for web frontends
- Batch processing capabilities
- File upload support for text synthesis

### Features
- `/api/v1/synthesize` - Main TTS endpoint
- `/api/v1/health` - Health status monitoring
- `/api/v1/languages` - Available languages listing
- `/api/v1/synthesize/file` - File-based synthesis

### Documentation
- Interactive Swagger UI at `/docs`
- ReDoc documentation at `/redoc`
- OpenAPI specification at `/openapi.json`
- Comprehensive README with examples
- Python client with CLI interface

### DevOps
- Docker image with multi-stage build
- Docker Compose configuration
- Automated build script
- Health checks configuration
- Environment variables support

## [Unreleased]

### Planned
- Streaming audio generation
- Additional language support
- Voice cloning capabilities
- Real-time synthesis optimization
- Kubernetes deployment manifests
- Prometheus metrics endpoint
- Rate limiting
- API authentication
