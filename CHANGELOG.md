# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
