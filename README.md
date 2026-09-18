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
| `FunAudioLLM/SenseVoiceSmall` | ASR + Emotion + Audio events | ~900MB |
| `funasr/fsmn-vad` | Voice activity detection | ~2MB |
| `facebook/nllb-200-distilled-600M` | Translation (200+ languages) | ~2.4GB |
| `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` | Emotion Detection (legacy path) | ~1.9GB |
| `openai/whisper-base` | Speech-to-Text (legacy path) | ~140MB |

All models are vendored in-repo under `models/` via Git LFS, so the app runs on machines
with no HuggingFace access. See [Running without HuggingFace access](#running-without-huggingface-access).

## Running without HuggingFace access

All models are already committed to this repo under `models/` (~5.8GB via Git LFS), so a
clone is all you need. Models are loaded from `models/` when present, and only fall back to
downloading from HuggingFace when one is missing.

### Setup on a restricted machine

```bash
# 1. Install Git LFS FIRST - before cloning
git lfs install

# 2. Clone; the model weights come down with it
git clone https://github.com/manishmitra017/voice_guard_rail.git
cd voice_guard_rail

# 3. Dependencies still come from PyPI, not HuggingFace
uv sync
cd frontend && npm install && cd ..

# 4. Start in offline mode
VG_OFFLINE=1 ./start-local.sh
```

If you cloned *before* running `git lfs install`, the weights arrive as small text pointers
instead of real files. Fix that with `git lfs pull`.

`VG_OFFLINE=1` sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` and makes a missing model raise
a clear error immediately, instead of hanging until a blocked connection times out. Use it on
a restricted network so mistakes surface in seconds rather than minutes.

### Options

| Variable | Purpose |
|----------|---------|
| `VG_MODELS_DIR` | Load models from elsewhere, e.g. a USB drive or network share |
| `VG_OFFLINE=1` | Never contact HuggingFace; fail fast if a model is missing |

### Refreshing the vendored models

On a machine **with** internet:

```bash
python scripts/download_models.py     # re-fetch + reshard into models/
git add models && git commit -m "Refresh vendored models"
git push
```

### Notes and limits

- **Sharded checkpoints.** GitHub rejects any single Git LFS file over 2GiB, so the two large
  checkpoints are stored as 3 safetensors shards each (~1GB max). `from_pretrained()`
  reassembles them via the generated index file, with no change in precision.
  `scripts/download_models.py` reshards automatically.
- **LFS bandwidth.** Each full clone pulls ~5.8GB against the account's monthly Git LFS
  allowance. On an existing clone, `git lfs pull` is much cheaper than re-cloning.
- **No-Git alternative.** If Git LFS is unavailable or over quota, copy the `models/`
  directory to the target machine by USB or network share and point `VG_MODELS_DIR` at it.
  No code changes are needed.
- **SenseVoice + PyPI.** `funasr` pip-installs a model's `requirements.txt` at load time, so
  the vendored copy deliberately omits that file. This needs PyPI, never HuggingFace.

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
- **Disk**: ~6GB for models (already included in the clone)

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

> **Note:** This repo stores all model weights under `models/` via [Git LFS](https://git-lfs.com)
> (~5.8GB). Install `git-lfs` **before** cloning, or the weights arrive as small text
> pointers instead of real files.

```bash
# Install Git LFS (once per machine)
brew install git-lfs        # macOS; apt-get install git-lfs on Debian/Ubuntu
git lfs install

# Clone the repository
git clone https://github.com/manishmitra017/voice_guard_rail.git
cd voice_guard_rail

# If you cloned before installing git-lfs, fetch the weights now
git lfs pull

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
