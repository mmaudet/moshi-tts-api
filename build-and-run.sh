#!/bin/bash

# Build and run script for Moshi TTS API

set -e

echo "🚀 Moshi TTS API Docker Setup"
echo "=============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for NVIDIA Docker support (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    GPU_SUPPORT="--gpus all"
else
    echo "⚠️  No NVIDIA GPU detected, running in CPU mode"
    GPU_SUPPORT=""
fi

# Build the Docker image
echo ""
echo "📦 Building Docker image..."
docker build -t moshi-tts-api:latest .

# Create models directory if it doesn't exist
mkdir -p models

# Stop and remove existing container if it exists
docker stop moshi-tts-api 2>/dev/null || true
docker rm moshi-tts-api 2>/dev/null || true

# Run the container
echo ""
echo "🏃 Starting container..."
docker run -d \
    --name moshi-tts-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    $GPU_SUPPORT \
    --restart unless-stopped \
    moshi-tts-api:latest

echo ""
echo "✅ Container started successfully!"
echo ""
echo "📝 API Documentation: http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
echo ""
echo "📌 Example usage:"
echo "  curl -X POST http://localhost:8000/tts \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"text\": \"Bonjour, ceci est un test de synthèse vocale\"}' \\"
echo "       --output test.wav"
echo ""
echo "📊 View logs:"
echo "  docker logs -f moshi-tts-api"
echo ""
echo "🛑 Stop container:"
echo "  docker stop moshi-tts-api"
