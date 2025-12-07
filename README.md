# Voice Emotion Detector

Real-time speech emotion recognition from your microphone using AI. Detects **7 emotions** with **92% accuracy** and transcribes what you said.

## Features

- **7 Emotion Detection**: Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised
- **Speech-to-Text**: Transcribes your speech using OpenAI Whisper
- **Real-time Analysis**: Record from your microphone with start/stop control
- **Web Interface**: Clean Streamlit UI with visual emotion display
- **High Accuracy**: 92% accuracy using Whisper-based emotion model

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VOICE EMOTION DETECTOR                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐                                                       │
│   │  🎤 Record   │  Click "Start Recording" → Speak → Click "Stop"      │
│   │  (Mic Input) │                                                       │
│   └──────┬───────┘                                                       │
│          │                                                               │
│          │  16kHz WAV audio                                              │
│          ▼                                                               │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                    AUDIO PROCESSING                           │      │
│   ├──────────────────────┬───────────────────────────────────────┤      │
│   │                      │                                        │      │
│   │   ┌──────────────┐   │   ┌────────────────────────────────┐  │      │
│   │   │   Whisper    │   │   │  Whisper Emotion Classifier    │  │      │
│   │   │   (base)     │   │   │  (large-v3 fine-tuned)         │  │      │
│   │   │              │   │   │                                 │  │      │
│   │   │  Speech to   │   │   │  Audio → Mel Spectrogram →     │  │      │
│   │   │  Text        │   │   │  Transformer → 7 Emotions      │  │      │
│   │   └──────┬───────┘   │   └──────────────┬─────────────────┘  │      │
│   │          │           │                  │                     │      │
│   │          ▼           │                  ▼                     │      │
│   │   ┌──────────────┐   │   ┌────────────────────────────────┐  │      │
│   │   │ "Hello, how  │   │   │  😊 Happy (85.2%)              │  │      │
│   │   │  are you?"   │   │   │  😐 Neutral (10.1%)            │  │      │
│   │   └──────────────┘   │   │  😢 Sad (2.3%)                 │  │      │
│   │                      │   │  ...                            │  │      │
│   └──────────────────────┴───┴────────────────────────────────┘  │      │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                      STREAMLIT UI                             │      │
│   │  ┌─────────────────────────────────────────────────────────┐ │      │
│   │  │  📝 What You Said: "Hello, how are you?"                │ │      │
│   │  │  🎭 Detected Emotion: 😊 Happy (85.2% confidence)       │ │      │
│   │  │  📊 All Scores: [bar chart of 7 emotions]               │ │      │
│   │  └─────────────────────────────────────────────────────────┘ │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Models Used

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| `openai/whisper-base` | Speech-to-Text | ~140MB | [OpenAI Whisper](https://github.com/openai/whisper) |
| `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` | Emotion Detection | ~600MB | [HuggingFace](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) |

## Detected Emotions

| Emotion | Emoji | Color |
|---------|-------|-------|
| Angry | 😠 | Red |
| Disgust | 🤢 | Green |
| Fearful | 😨 | Purple |
| Happy | 😊 | Green |
| Neutral | 😐 | Gray |
| Sad | 😢 | Blue |
| Surprised | 😲 | Yellow |

## Requirements

- **Python**: 3.10 or higher
- **OS**: macOS (tested), Linux, Windows
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: ~2GB for model downloads
- **Microphone**: Built-in or external

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/manishmitra017/voice_guard_rail.git
cd voice_guard_rail
```

### 2. Install uv (Python Package Manager)

If you don't have `uv` installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install Dependencies

```bash
uv sync
```

This will:
- Create a virtual environment automatically
- Install all Python dependencies
- Download required packages (~500MB)

### 4. Run the App

```bash
./start.sh
```

Or manually:

```bash
uv run streamlit run app.py
```

### 5. Open in Browser

Navigate to: **http://localhost:8501**

## First Run

On first run, the app will download AI models:

1. **Whisper base model** (~140MB) - for speech transcription
2. **Emotion classifier** (~600MB) - for emotion detection

This happens once and models are cached for future runs.

## Usage

1. **Click "Start Recording"** - begins capturing audio from your microphone
2. **Speak** - say something with emotion!
3. **Click "Stop Recording"** - stops recording and analyzes
4. **View Results**:
   - 📝 **What You Said** - transcribed text
   - 🎭 **Detected Emotion** - primary emotion with confidence
   - 📊 **All Emotion Scores** - probability distribution

## Project Structure

```
voice_guard_rail/
├── app.py                      # Local Streamlit app (uses sounddevice)
├── app_cloud.py                # Cloud Streamlit app (browser-based recording)
├── start.sh                    # Quick start script (local)
├── deploy.sh                   # AWS deployment script
├── Dockerfile                  # Container image for deployment
├── pyproject.toml              # Python dependencies (uv)
├── README.md                   # This file
├── cdk/                        # AWS CDK infrastructure
│   ├── bin/app.ts              # CDK app entry point
│   └── lib/voice-emotion-stack.ts  # EC2 Spot + networking
└── src/
    ├── __init__.py
    ├── audio_recorder.py       # Microphone recording (local only)
    ├── emotion_classifier.py   # 7-emotion Whisper model
    └── speech_transcriber.py   # Whisper speech-to-text
```

## Technical Details

### Audio Recording (`audio_recorder.py`)
- Uses `sounddevice` for cross-platform microphone access
- Records at 16kHz mono (required by models)
- Saves to temporary WAV file for processing

### Speech Transcription (`speech_transcriber.py`)
- Uses OpenAI Whisper (base model)
- Loads audio with `librosa` (no ffmpeg dependency)
- Returns transcribed text and detected language

### Emotion Classification (`emotion_classifier.py`)
- Uses Whisper-large-v3 fine-tuned on emotion datasets
- Trained on: RAVDESS, SAVEE, TESS, URDU datasets
- Returns 7 emotion probabilities with confidence scores

## Troubleshooting

### Microphone Access (macOS)

If the app can't access your microphone:

1. Go to **System Preferences** → **Security & Privacy** → **Privacy** → **Microphone**
2. Enable access for **Terminal** or your IDE

### Microphone Access (Linux)

```bash
# Check if your user is in the audio group
groups $USER

# Add to audio group if needed
sudo usermod -a -G audio $USER
```

### Model Download Issues

If models fail to download:

```bash
# Clear HuggingFace cache and retry
rm -rf ~/.cache/huggingface
uv run streamlit run app.py
```

### Port Already in Use

```bash
# Use a different port
uv run streamlit run app.py --server.port 8502
```

## Development

### Install Dev Dependencies

```bash
uv sync --dev
```

### Run Tests

```bash
uv run pytest
```

### Lint Code

```bash
uv run ruff check .
```

## AWS Deployment

Deploy to AWS EC2 Spot instances for **~$12-15/month**.

### Prerequisites

1. AWS CLI configured with credentials
2. Node.js 18+ (for CDK)
3. AWS CDK CLI: `npm install -g aws-cdk`

### Quick Deploy

```bash
./deploy.sh
```

### Manual Deploy

```bash
# Navigate to CDK directory
cd cdk

# Install dependencies
npm install

# Deploy to AWS
npm run deploy
```

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    AWS Cloud (~$12-15/month)              │
├──────────────────────────────────────────────────────────┤
│                                                           │
│   ┌─────────────────┐         ┌────────────────────────┐ │
│   │   Your Domain   │────────▶│   EC2 Spot t3.medium   │ │
│   │  (optional)     │         │   - 4GB RAM + 4GB swap │ │
│   └─────────────────┘         │   - nginx reverse proxy│ │
│                               │   - Streamlit :8501    │ │
│                               │   - Auto-restart       │ │
│                               └────────────────────────┘ │
│                                           │              │
│                               ┌───────────▼────────────┐ │
│                               │   ML Models (cached)   │ │
│                               │   - Whisper base       │ │
│                               │   - Emotion classifier │ │
│                               └────────────────────────┘ │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Cost Breakdown

| Resource | Monthly Cost |
|----------|-------------|
| EC2 Spot t3.medium | ~$10 |
| EBS Storage (20GB) | ~$2 |
| Data Transfer | ~$1 |
| **Total** | **~$12-15** |

### Cloud vs Local App

| Feature | `app.py` (Local) | `app_cloud.py` (Cloud) |
|---------|------------------|------------------------|
| Audio Recording | Server microphone (sounddevice) | Browser microphone (WebRTC) |
| Deployment | Local machine only | AWS, Docker, any cloud |
| HTTPS Required | No | Yes (for microphone access) |

### SSH Access

```bash
ssh -i <your-key.pem> ec2-user@<elastic-ip>

# View logs
sudo journalctl -u voice-emotion -f

# Restart service
sudo systemctl restart voice-emotion
```

### Destroy Stack

```bash
cd cdk && npm run destroy
```

## Future Improvements

- [x] AWS deployment (EC2 Spot)
- [ ] Real-time streaming analysis
- [ ] Emotion history tracking
- [ ] Multiple language support
- [ ] Custom emotion model training

## License

MIT License

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [HuggingFace](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) - Emotion model
- [Streamlit](https://streamlit.io/) - Web interface
