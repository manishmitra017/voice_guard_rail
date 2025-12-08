"""
FastAPI backend for Voice Emotion Detection.
Exposes REST endpoints for audio analysis.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.emotion_classifier import EmotionClassifier
from src.speech_transcriber import SpeechTranscriber


# Global model instances
emotion_classifier: Optional[EmotionClassifier] = None
speech_transcriber: Optional[SpeechTranscriber] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global emotion_classifier, speech_transcriber

    print("Loading models... (this may take a minute on first run)")

    # Initialize and load models
    emotion_classifier = EmotionClassifier(use_gpu=False)
    emotion_classifier.load_model()
    print("Emotion classifier loaded")

    speech_transcriber = SpeechTranscriber(model_size="base")
    speech_transcriber.load_model()
    print("Speech transcriber loaded")

    print("All models loaded! Server ready.")

    yield

    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="Voice Emotion Detector API",
    description="Real-time speech emotion recognition using AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class EmotionResult(BaseModel):
    emotion: str
    emoji: str
    color: str
    confidence: float
    raw_label: str
    all_probabilities: dict


class TranscriptionResult(BaseModel):
    text: str
    language: str


class AnalyzeResponse(BaseModel):
    transcription: TranscriptionResult
    emotion: EmotionResult


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and models are loaded."""
    return HealthResponse(
        status="healthy",
        models_loaded=(
            emotion_classifier is not None
            and emotion_classifier.is_loaded()
            and speech_transcriber is not None
            and speech_transcriber.is_loaded()
        )
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze uploaded audio file for emotion and transcription.

    Accepts: WAV, MP3, WebM, OGG audio files
    Returns: Emotion classification and transcription
    """
    if emotion_classifier is None or speech_transcriber is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/webm", "audio/ogg", "audio/x-wav"]
    content_type = audio.content_type or ""

    # Be lenient with content types
    if not any(t in content_type for t in ["audio", "octet-stream"]):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Expected audio file."
        )

    # Save to temporary file
    suffix = ".wav"
    if "webm" in content_type:
        suffix = ".webm"
    elif "ogg" in content_type:
        suffix = ".ogg"
    elif "mpeg" in content_type or "mp3" in content_type:
        suffix = ".mp3"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Analyze audio
        transcription = speech_transcriber.transcribe(tmp_path)
        emotion = emotion_classifier.classify(tmp_path)

        return AnalyzeResponse(
            transcription=TranscriptionResult(**transcription),
            emotion=EmotionResult(**emotion)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Clean up temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.get("/emotions")
async def list_emotions():
    """List all detectable emotions with their display info."""
    return EmotionClassifier.EMOTION_DISPLAY


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
