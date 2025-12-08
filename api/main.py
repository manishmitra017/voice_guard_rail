"""
FastAPI backend for Voice Emotion Detection.
Exposes REST endpoints for audio analysis with multilingual support.

Features:
- SenseVoice: Unified ASR + Emotion Recognition (50+ languages)
- NLLB Translation: 200+ language translation
- Legacy Whisper mode for backward compatibility
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.sensevoice_analyzer import SenseVoiceAnalyzer, get_sensevoice_analyzer
from src.translator import TranslationService, get_translator

# Legacy imports for backward compatibility
from src.emotion_classifier import EmotionClassifier
from src.speech_transcriber import SpeechTranscriber


# Configuration
USE_SENSEVOICE = os.getenv("USE_SENSEVOICE", "true").lower() == "true"
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "true").lower() == "true"
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"


# Global model instances
sensevoice_analyzer: Optional[SenseVoiceAnalyzer] = None
translator: Optional[TranslationService] = None

# Legacy instances
emotion_classifier: Optional[EmotionClassifier] = None
speech_transcriber: Optional[SpeechTranscriber] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global sensevoice_analyzer, translator, emotion_classifier, speech_transcriber

    print("=" * 60)
    print("Loading models... (this may take a few minutes on first run)")
    print("=" * 60)

    if USE_SENSEVOICE:
        try:
            print("\n[1/2] Loading SenseVoice analyzer...")
            sensevoice_analyzer = get_sensevoice_analyzer(use_gpu=USE_GPU)
            sensevoice_analyzer.load_model()
            print("      SenseVoice loaded successfully!")
        except Exception as e:
            print(f"      Failed to load SenseVoice: {e}")
            print("      Falling back to legacy Whisper mode...")
            USE_SENSEVOICE_RUNTIME = False
    else:
        print("\n[1/2] Loading legacy Whisper models...")
        emotion_classifier = EmotionClassifier(use_gpu=USE_GPU)
        emotion_classifier.load_model()
        print("      Emotion classifier loaded")

        speech_transcriber = SpeechTranscriber(model_size="base")
        speech_transcriber.load_model()
        print("      Speech transcriber loaded")

    if ENABLE_TRANSLATION:
        try:
            print("\n[2/2] Loading translation model (NLLB-200)...")
            translator = get_translator(use_gpu=USE_GPU)
            translator.load_model()
            print("      Translation model loaded successfully!")
        except Exception as e:
            print(f"      Translation model loading deferred: {e}")
            print("      Translation will load on first use.")

    print("\n" + "=" * 60)
    print("All models loaded! Server ready.")
    print("=" * 60 + "\n")

    yield

    print("Shutting down...")


app = FastAPI(
    title="Voice Emotion Detector API",
    description="Real-time speech emotion recognition with multilingual translation",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    sensevoice_enabled: bool
    translation_enabled: bool


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
    language_name: Optional[str] = None


class TranslationItem(BaseModel):
    language_code: str
    language_name: str
    flag: str
    text: str


class AnalyzeResponse(BaseModel):
    transcription: TranscriptionResult
    emotion: EmotionResult
    audio_events: Optional[List[str]] = None
    translations: Optional[List[TranslationItem]] = None


class TranslateRequest(BaseModel):
    text: str
    source_language: str
    target_languages: List[str]


class TranslateResponse(BaseModel):
    source_text: str
    source_language: str
    translations: List[TranslationItem]


class LanguageInfo(BaseModel):
    code: str
    name: str
    flag: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    models_loaded = False

    if USE_SENSEVOICE and sensevoice_analyzer:
        models_loaded = sensevoice_analyzer.is_loaded()
    elif emotion_classifier and speech_transcriber:
        models_loaded = emotion_classifier.is_loaded() and speech_transcriber.is_loaded()

    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        sensevoice_enabled=USE_SENSEVOICE and sensevoice_analyzer is not None,
        translation_enabled=ENABLE_TRANSLATION and translator is not None,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    audio: UploadFile = File(...),
    language: str = Query("auto", description="Language code: auto, en, zh, ja, ko, yue"),
    translate_to: Optional[str] = Query(None, description="Comma-separated target languages: es,fr,de"),
):
    """
    Analyze audio for transcription, emotion, and optionally translate.

    - **audio**: Audio file (WAV, MP3, WebM, OGG)
    - **language**: Source language for transcription (auto-detect by default)
    - **translate_to**: Optional comma-separated list of target languages
    """
    # Validate content type
    content_type = audio.content_type or ""
    if not any(t in content_type for t in ["audio", "octet-stream"]):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Expected audio file."
        )

    # Determine file suffix
    suffix = ".wav"
    if "webm" in content_type:
        suffix = ".webm"
    elif "ogg" in content_type:
        suffix = ".ogg"
    elif "mpeg" in content_type or "mp3" in content_type:
        suffix = ".mp3"

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Analyze with SenseVoice or legacy mode
        if USE_SENSEVOICE and sensevoice_analyzer and sensevoice_analyzer.is_loaded():
            result = sensevoice_analyzer.analyze(tmp_path, language=language)

            transcription = TranscriptionResult(
                text=result["transcription"]["text"],
                language=result["transcription"]["language"],
                language_name=result["transcription"].get("language_name"),
            )
            emotion = EmotionResult(**result["emotion"])
            audio_events = result.get("audio_events", [])
        else:
            # Legacy Whisper mode
            if not emotion_classifier or not speech_transcriber:
                raise HTTPException(status_code=503, detail="Models not loaded")

            trans_result = speech_transcriber.transcribe(tmp_path)
            emo_result = emotion_classifier.classify(tmp_path)

            transcription = TranscriptionResult(
                text=trans_result["text"],
                language=trans_result["language"],
            )
            emotion = EmotionResult(**emo_result)
            audio_events = None

        # Handle translations
        translations = None
        if translate_to and translator:
            target_langs = [lang.strip() for lang in translate_to.split(",")]
            translations = await _translate_text(
                transcription.text,
                transcription.language,
                target_langs
            )

        return AnalyzeResponse(
            transcription=transcription,
            emotion=emotion,
            audio_events=audio_events,
            translations=translations,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Translate text to multiple target languages.

    - **text**: Text to translate
    - **source_language**: Source language code
    - **target_languages**: List of target language codes
    """
    if not translator:
        raise HTTPException(status_code=503, detail="Translation service not available")

    if not translator.is_loaded():
        try:
            translator.load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load translator: {e}")

    translations = await _translate_text(
        request.text,
        request.source_language,
        request.target_languages
    )

    return TranslateResponse(
        source_text=request.text,
        source_language=request.source_language,
        translations=translations,
    )


async def _translate_text(
    text: str,
    source_lang: str,
    target_langs: List[str]
) -> List[TranslationItem]:
    """Helper to translate text to multiple languages."""
    if not translator or not text.strip():
        return []

    if not translator.is_loaded():
        translator.load_model()

    translations = []
    lang_info = {l["code"]: l for l in translator.get_supported_languages()}

    for target_lang in target_langs:
        if target_lang == source_lang:
            continue

        try:
            result = translator.translate(text, source_lang, target_lang)
            info = lang_info.get(target_lang, {})

            translations.append(TranslationItem(
                language_code=target_lang,
                language_name=info.get("name", target_lang),
                flag=info.get("flag", "🏳️"),
                text=result.translated_text,
            ))
        except Exception as e:
            print(f"Translation to {target_lang} failed: {e}")

    return translations


@app.get("/languages", response_model=List[LanguageInfo])
async def list_languages():
    """List all supported languages for translation."""
    if translator:
        langs = translator.get_supported_languages()
        return [LanguageInfo(**lang) for lang in langs]

    # Return basic list if translator not loaded
    return [
        LanguageInfo(code="en", name="English", flag="🇬🇧"),
        LanguageInfo(code="es", name="Spanish", flag="🇪🇸"),
        LanguageInfo(code="fr", name="French", flag="🇫🇷"),
        LanguageInfo(code="de", name="German", flag="🇩🇪"),
        LanguageInfo(code="zh", name="Chinese", flag="🇨🇳"),
        LanguageInfo(code="ja", name="Japanese", flag="🇯🇵"),
        LanguageInfo(code="ko", name="Korean", flag="🇰🇷"),
    ]


@app.get("/emotions")
async def list_emotions():
    """List all detectable emotions with display info."""
    if USE_SENSEVOICE and sensevoice_analyzer:
        return sensevoice_analyzer.EMOTION_DISPLAY
    return EmotionClassifier.EMOTION_DISPLAY


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
