#!/bin/bash
# Fix permissions on models directory for Docker container
# This script fixes permission issues when models are downloaded by HuggingFace Hub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

echo "=========================================="
echo "Fixing Model Permissions"
echo "=========================================="
echo ""

if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Models directory not found: $MODELS_DIR"
    echo "   Models will be downloaded on first run."
    exit 0
fi

echo "📂 Models directory: $MODELS_DIR"
echo "🔧 Setting ownership to UID 1001 (appuser)..."
echo ""

# Change ownership to UID 1001 (appuser in container)
sudo chown -R 1001:1001 "$MODELS_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Permissions fixed successfully!"
    echo ""
    echo "You can now restart the container:"
    echo "  docker compose restart"
else
    echo ""
    echo "❌ Failed to fix permissions"
    echo "   Make sure you have sudo access"
    exit 1
fi
