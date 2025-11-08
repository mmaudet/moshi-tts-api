# 🎙️ Moshi TTS API

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

REST API for text-to-speech synthesis using [Moshi model from Kyutai Labs](https://github.com/kyutai-labs/moshi), with interactive Swagger documentation and Docker deployment.

## ✨ Features

- 🌐 **Bilingual Support**: French and English
- 🎤 **Voice Selection**: 10+ VCTK voices with customizable options
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

### Installation

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

### With Docker Compose

```bash
# With GPU (default)
docker-compose up -d

# Without GPU (edit docker-compose.yml to remove the deploy section)
docker-compose up -d
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
curl -X POST http://localhost:8000/api/v1/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Bonjour, je suis Moshi, votre assistant vocal.",
       "language": "fr"
     }' \
     --output bonjour.wav
```

#### English Synthesis
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, I am Moshi, your voice assistant.",
       "language": "en"
     }' \
     --output hello.wav
```

#### With Voice Selection
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
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
curl -X POST http://localhost:8000/api/v1/synthesize \
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

#### 5. **POST /api/v1/synthesize** - Voice Generation
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
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

#### 6. **POST /api/v1/synthesize/file** - Synthesis from File
```bash
curl -X POST http://localhost:8000/api/v1/synthesize/file \
     -F "file=@my_text.txt" \
     -F "language=fr" \
     --output audio.wav
```

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Specify which GPU to use
docker run -e CUDA_VISIBLE_DEVICES=0 ...

# Change model cache directory
docker run -e HF_HOME=/custom/path ...

# Disable transformers cache
docker run -e TRANSFORMERS_OFFLINE=1 ...
```

### Model Customization

Edit `app_final.py` to change the model:
```python
DEFAULT_TTS_REPO = "kyutai/tts-1.6b-en_fr"  # or another model
DEFAULT_VOICE_REPO = "kyutai/tts-voices"
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
    "http://localhost:8000/api/v1/synthesize",
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

axios.post('http://localhost:8000/api/v1/synthesize', {
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
- URL: http://localhost:8000/api/v1/synthesize
- Body: JSON with `{"text": "your text", "language": "en"}`
- Response Format: File

## 🎤 Voice Presets

The API includes multiple voice presets from the VCTK corpus:

- `default` - Default voice (vctk/p225_023.wav)
- `vctk_p225` through `vctk_p234` - Various VCTK voices

You can list all available voices using the `/api/v1/voices` endpoint.

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

## 📸 Screenshots

### Swagger UI
The interactive documentation allows you to test all endpoints directly from the browser:

- `/docs` - Swagger UI interface
- `/redoc` - Alternative ReDoc documentation
- `/openapi.json` - OpenAPI specification

## 🙏 Acknowledgments

- [Kyutai Labs](https://github.com/kyutai-labs) for the Moshi model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Docker](https://www.docker.com/) for containerization

## 📧 Contact

For any questions or suggestions, feel free to open an issue on GitHub.

---

⭐ If this project is useful to you, don't forget to give it a star on GitHub!
