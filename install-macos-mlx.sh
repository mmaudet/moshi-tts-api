#!/bin/bash
#
# Moshi TTS API - Native macOS Installation with MLX
# Optimized for Apple Silicon (M1/M2/M3/M4/M5)
#
# This script installs and runs Moshi TTS API natively on macOS using MLX framework
# for optimal performance with Metal GPU acceleration.
#

set -e

echo "🍎 Moshi TTS API - macOS MLX Installation"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script is for macOS only"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "⚠️  Warning: MLX is optimized for Apple Silicon (ARM64)"
    echo "   You are running on: $ARCH"
    echo "   Performance may be suboptimal"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10 or later"
    echo "   Visit: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="venv-moshi-mlx"
if [ -d "$VENV_DIR" ]; then
    echo "📁 Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install moshi-mlx and dependencies
echo "📥 Installing moshi-mlx (this may take a few minutes)..."
pip install moshi-mlx

echo "📥 Installing FastAPI and dependencies..."
pip install \
    'fastapi>=0.104.1' \
    'uvicorn[standard]>=0.24.0' \
    'pydantic>=2.4.2' \
    'pydantic-settings>=2.1.0' \
    'numpy>=1.26.0' \
    'scipy>=1.11.4' \
    'python-multipart>=0.0.6' \
    'aiofiles>=23.2.1'

# Create models directory
mkdir -p models

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "3. Access the API:"
echo "   http://localhost:8000/docs"
echo ""
echo "🚀 For production, use:"
echo "   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1"
echo ""
echo "💡 Note: The first run will download the Moshi model (~800MB)"
echo "   This will be cached in ./models/ for future use"
echo ""
