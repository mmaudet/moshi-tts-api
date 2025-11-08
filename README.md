# 🎙️ Moshi TTS API

[![Docker Hub](https://img.shields.io/docker/v/mmaudet/moshi-tts-api?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/mmaudet/moshi-tts-api)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

REST API for text-to-speech synthesis using [Moshi model from Kyutai Labs](https://github.com/kyutai-labs/moshi), with interactive Swagger documentation and Docker deployment.

## ✨ Features

- 🌐 **Bilingual Support**: French and English
- 🎤 **44 Voice Presets**: VCTK, CML-TTS French, Expresso emotions, EARS speakers
- 🎭 **Emotional Speech**: Happy, angry, calm, confused, whisper, and more
- 📖 **Swagger Documentation**: Interactive interface to test the API
- 🎵 **High-Quality Audio**: 24kHz in WAV or RAW format
- 🚀 **GPU Support**: Automatic CUDA acceleration
- 🔒 **Secure**: Non-root user, input validation
- 📦 **Docker**: Simple and reproducible deployment
- 🔄 **RESTful API**: Well-structured endpoints with OpenAPI
- 📊 **Health Checks**: Service status monitoring

## 🚀 Quick Start

### Prerequisites
- Docker installed
- NVIDIA Docker Runtime (optional, for GPU support)
- At least 8GB RAM
- ~10GB disk space for the model

### Option 1: Using Pre-built Image (Recommended ⚡)

The fastest way to get started! No need to clone or build.

#### GPU Version (Linux with NVIDIA GPU)

```bash
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v moshi-models:/app/models \
    --gpus all \
    mmaudet/moshi-tts-api:latest
```

#### CPU Version (Mac, Windows, or Linux without GPU)

**For Mac with Apple Silicon (M1/M2/M3/M4/M5):**
```bash
# The image is AMD64 only, but runs on Apple Silicon via Rosetta 2 emulation
# Docker Desktop automatically handles this - just run:
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v moshi-models:/app/models \
    --platform linux/amd64 \
    mmaudet/moshi-tts-api:cpu
```

**For Intel Mac, Windows, or Linux without GPU:**
```bash
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v moshi-models:/app/models \
    mmaudet/moshi-tts-api:cpu
```

**Note**:
- CPU version is slower than GPU but works on all platforms
- Mac ARM64 users: The `--platform linux/amd64` flag uses Rosetta 2 emulation
- Performance on Mac M1-M5 is still good thanks to Rosetta 2

Access the API at: http://localhost:8000/docs

### Option 2: Build from Source

1. **Clone the project**
```bash
git clone https://github.com/mmaudet/moshi-tts-api.git
cd moshi-tts-api
```

2. **Quick build and launch**
```bash
chmod +x build-and-run.sh
./build-and-run.sh
```

Or manually:

```bash
# Build
docker build -t moshi-tts-api:latest .

# Run with GPU
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    --gpus all \
    moshi-tts-api:latest

# Run without GPU (CPU only)
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    moshi-tts-api:latest
```

### Option 3: With Docker Compose

**Using pre-built image** (update `docker-compose.yml`):
```yaml
services:
  moshi-tts-api:
    image: mmaudet/moshi-tts-api:latest
    # Remove the 'build: .' line
```

**Building from source**:
```bash
# With GPU (default)
docker compose up -d

# Without GPU (edit docker-compose.yml to remove the deploy section)
docker compose up -d
```

## 📖 Usage

### Interactive Documentation (Swagger)

Once the API is started, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Quick Test with Script
```bash
chmod +x test_api.sh
./test_api.sh
```

### Usage Examples with cURL

#### French Synthesis
```bash
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Bonjour, je suis Moshi, votre assistant vocal.",
       "language": "fr"
     }' \
     --output bonjour.wav
```

#### English Synthesis
```bash
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, I am Moshi, your voice assistant.",
       "language": "en"
     }' \
     --output hello.wav
```

#### With Voice Selection
```bash
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello with a different voice.",
       "language": "en",
       "voice": "vctk_p226"
     }' \
     --output custom_voice.wav
```

#### RAW Format (PCM)
```bash
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Test audio",
       "language": "en",
       "format": "raw"
     }' \
     --output test.raw

# Convert RAW to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i test.raw output.wav
```

### Available Endpoints

#### 1. **GET /** - API Information
```bash
curl http://localhost:8000/
```

#### 2. **GET /api/v1/health** - Health Status
```bash
curl http://localhost:8000/api/v1/health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "available_languages": ["fr", "en"],
  "api_version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 3. **GET /api/v1/languages** - Available Languages
```bash
curl http://localhost:8000/api/v1/languages
```
Response:
```json
{
  "languages": [
    {"code": "fr", "name": "French (Français)"},
    {"code": "en", "name": "English"}
  ]
}
```

#### 4. **GET /api/v1/voices** - Available Voices
```bash
curl http://localhost:8000/api/v1/voices
```
Response:
```json
{
  "voices": [
    {"id": "default", "name": "vctk_p225", "description": "Default voice"},
    {"id": "vctk_p225", "name": "vctk_p225", "description": "VCTK voice p225"},
    {"id": "vctk_p226", "name": "vctk_p226", "description": "VCTK voice p226"}
  ]
}
```

#### 5. **POST /api/v1/tts** - Text-to-Speech Generation
```bash
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your text here",
       "language": "fr",
       "format": "wav",
       "voice": "default"
     }' \
     --output audio.wav
```

Parameters:
- `text` (required): Text to synthesize (1-5000 characters)
- `language` (optional, default: "fr"): Language code ("fr" or "en")
- `format` (optional, default: "wav"): Output format ("wav" or "raw")
- `voice` (optional, default: "default"): Voice preset to use

#### 6. **POST /api/v1/tts/file** - Text-to-Speech from File
```bash
curl -X POST http://localhost:8000/api/v1/tts/file \
     -F "file=@my_text.txt" \
     -F "language=fr" \
     --output audio.wav
```

## 🔧 Advanced Configuration

### Configuration Management

The API uses **pydantic-settings** for type-safe configuration management. Configuration can be set via:

1. **`.env` file** (local development)
2. **Environment variables** (Docker/production)
3. **Default values** (fallback)

#### Setup for Local Development

```bash
# Copy the template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Important**: The `.env` file is gitignored and should never be committed!

#### Available Configuration Options

See `.env.example` for all available settings:

```bash
# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
WORKERS=1

# Model Configuration
DEFAULT_TTS_REPO=kyutai/tts-1.6b-en_fr
DEFAULT_VOICE_REPO=kyutai/tts-voices
SAMPLE_RATE=24000
MODEL_DEVICE=cuda  # or cpu, auto-detected if not set
MODEL_DTYPE=auto   # auto, bfloat16, or float32
MODEL_N_Q=32       # Number of codebooks
MODEL_TEMP=0.6     # Temperature for generation
MODEL_CFG_COEF=2.0 # CFG coefficient

# CORS
CORS_ORIGINS=*  # Change in production!
CORS_CREDENTIALS=true

# Environment
ENVIRONMENT=production
DEBUG=false
```

#### Docker Configuration

With Docker, set environment variables in `docker-compose.yml`:

```yaml
environment:
  - DEFAULT_TTS_REPO=kyutai/tts-1.6b-en_fr
  - SAMPLE_RATE=24000
  - LOG_LEVEL=debug
```

Or pass them directly:

```bash
docker run -e LOG_LEVEL=debug -e MODEL_DEVICE=cpu ...
```

### Performance

- **GPU**: Real-time or faster generation
- **CPU**: Slower generation (2-10x real-time depending on CPU)
- **Memory**: ~6GB for the model in bf16
- **First Request**: Slower (model loading)

## 🐳 Useful Docker Commands

```bash
# View logs
docker logs -f moshi-tts-api

# Stop container
docker stop moshi-tts-api

# Restart
docker restart moshi-tts-api

# Remove container
docker rm -f moshi-tts-api

# Clean image
docker rmi moshi-tts-api:latest

# Enter container
docker exec -it moshi-tts-api bash
```

## 🔍 Debugging

### API Doesn't Start
```bash
# Check logs
docker logs moshi-tts-api

# Check if port 8000 is free
lsof -i :8000
```

### GPU Error
```bash
# Verify NVIDIA Docker
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Insufficient Memory
- Use a smaller model
- Increase Docker memory
- Use CPU mode

## 📦 Multi-Architecture Build

To create an ARM64 and AMD64 compatible image:
```bash
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
    -t moshi-tts-api:latest --push .
```

## 🤝 Integration

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/tts",
    json={
        "text": "Hello world",
        "language": "en",
        "voice": "vctk_p225"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Node.js
```javascript
const axios = require('axios');
const fs = require('fs');

axios.post('http://localhost:8000/api/v1/tts', {
    text: 'Hello world',
    language: 'en',
    voice: 'default'
}, {
    responseType: 'arraybuffer'
}).then(response => {
    fs.writeFileSync('output.wav', response.data);
});
```

### n8n Integration
Use the HTTP Request node with:
- Method: POST
- URL: http://localhost:8000/api/v1/tts
- Body: JSON with `{"text": "your text", "language": "en"}`
- Response Format: File

## 🎤 Voice Presets

The API includes **44 voice presets** from multiple datasets in the [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) repository:

### Voice Collections

#### 1. VCTK Voices (English) - 10 voices
British English speakers from the Voice Cloning Toolkit:
- `vctk_p225` through `vctk_p234` - Various speaker characteristics
- Example: `"voice": "vctk/p226_023.wav"`

#### 2. CML-TTS French Voices (Français) - 10 voices
High-quality French speakers:
- `cml_fr_1406`, `cml_fr_1591`, `cml_fr_1770`, `cml_fr_2114`, `cml_fr_2154`
- `cml_fr_2216`, `cml_fr_2223`, `cml_fr_2465`, `cml_fr_296`, `cml_fr_3267`
- Example: `"voice": "cml-tts/fr/1406_1028_000009-0003.wav"`

#### 3. Expresso Voices (English with Emotions) - 9 voices
Emotional and stylistic variations:
- **Speaking Styles**: `default`, `enunciated`, `fast`, `projected`, `whisper`
- **Emotions**: `happy`, `angry`, `calm`, `confused`
- Example: `"voice": "expresso/ex03-ex01_happy_001_channel1_334s.wav"`

#### 4. EARS Voices (English) - 14 voices
Diverse English speakers (subset of 50 available):
- `ears_p001`, `ears_p002`, `ears_p003`, `ears_p004`, `ears_p005`
- `ears_p010`, `ears_p015`, `ears_p020`, `ears_p025`, `ears_p030`
- `ears_p035`, `ears_p040`, `ears_p045`, `ears_p050`
- Example: `"voice": "ears/p001/freeform_speech_01.wav"`

### Usage Examples

```bash
# English with emotional expression
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so happy today!", "language": "en", "voice": "expresso/ex03-ex01_happy_001_channel1_334s.wav"}' \
     --output happy_voice.wav

# French voice
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Bonjour, comment allez-vous?", "language": "fr", "voice": "cml-tts/fr/1406_1028_000009-0003.wav"}' \
     --output french_voice.wav

# Different English speaker
curl -X POST http://localhost:8000/api/v1/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a different voice.", "language": "en", "voice": "ears/p010/freeform_speech_01.wav"}' \
     --output ears_voice.wav
```

### List All Voices

You can list all available voices using the `/api/v1/voices` endpoint:

```bash
curl http://localhost:8000/api/v1/voices | jq
```

## 📄 License

This project uses Moshi from Kyutai Labs. See their [license](https://github.com/kyutai-labs/moshi/blob/main/LICENSE).

This API wrapper is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Kyutai Labs](https://github.com/kyutai-labs) for the Moshi model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Docker](https://www.docker.com/) for containerization

## 📧 Contact

For any questions or suggestions, feel free to open an issue on GitHub.

---

⭐ If this project is useful to you, don't forget to give it a star on GitHub!
