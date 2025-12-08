# Voice Emotion Detector

Real-time speech emotion recognition from your microphone using AI. Detects **7 emotions** with **92% accuracy** and transcribes what you said.

## Features

- **7 Emotion Detection**: Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised
- **Speech-to-Text**: Transcribes your speech using OpenAI Whisper
- **Real-time Analysis**: Record from your microphone with start/stop control
- **Modern Web UI**: React frontend with FastAPI backend
- **High Accuracy**: 92% accuracy using Whisper-based emotion model

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  React + FastAPI Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────┐         ┌─────────────────────────┐  │
│   │   React Frontend     │  HTTP   │   FastAPI Backend       │  │
│   │   (Vite + TypeScript)│────────▶│   (Python + uvicorn)    │  │
│   │                      │         │                         │  │
│   │   - MediaRecorder    │  POST   │   - POST /api/analyze   │  │
│   │   - WAV conversion   │────────▶│   - Emotion classifier  │  │
│   │   - Results display  │         │   - Speech transcriber  │  │
│   └──────────────────────┘         └─────────────────────────┘  │
│                                                                  │
│   Frontend: http://localhost:3000                                │
│   Backend:  http://localhost:8000                                │
└─────────────────────────────────────────────────────────────────┘
```

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `openai/whisper-base` | Speech-to-Text | ~140MB |
| `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` | Emotion Detection | ~600MB |

## Detected Emotions

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fearful | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprised | 😲 |

## Requirements

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **ffmpeg**: Required for audio processing
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: ~2GB for model downloads

### Installing ffmpeg

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt install ffmpeg
```

**Windows**:
```bash
# Using Chocolatey
choco install ffmpeg

# Or using Scoop
scoop install ffmpeg
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/manishmitra017/voice_guard_rail.git
cd voice_guard_rail

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start both servers
./start-local.sh
```

Open **http://localhost:3000** in your browser.

### Manual Start

```bash
# Terminal 1: Start FastAPI backend
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start React frontend
cd frontend && npm run dev
```

## Project Structure

```
voice_guard_rail/
├── api/                        # FastAPI backend
│   └── main.py                 # API endpoints
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.tsx             # Main React component
│   │   └── App.css             # Styles
│   ├── package.json
│   └── vite.config.ts
├── src/                        # Core ML modules
│   ├── emotion_classifier.py   # 7-emotion Whisper model
│   └── speech_transcriber.py   # Whisper speech-to-text
├── start-local.sh              # Local dev script
└── pyproject.toml              # Python dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/analyze` | POST | Analyze audio file (multipart form) |
| `/emotions` | GET | List all detectable emotions |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Analyze audio file
curl -X POST http://localhost:8000/analyze \
  -F "audio=@recording.wav"
```

## Troubleshooting

### Microphone Access

**macOS**: System Preferences → Security & Privacy → Privacy → Microphone → Enable for browser

**Linux**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
```

### Model Download Issues

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface
```

## License

MIT License
