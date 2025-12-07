#!/bin/bash
# Voice Emotion Detector - Start Script

set -e

echo "🎙️ Voice Emotion Detector"
echo "========================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Install it with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Navigate to project directory
cd "$(dirname "$0")"

# Sync dependencies (installs if needed)
echo "📦 Syncing dependencies..."
uv sync

# Run the app with file watcher disabled for cleaner logs
echo "🚀 Starting Streamlit app..."
echo "   Open http://localhost:8501 in your browser"
echo ""

# Disable file watcher to avoid SpeechBrain lazy module warnings
uv run streamlit run app.py \
    --server.headless=true \
    --server.fileWatcherType=none \
    --logger.level=error
