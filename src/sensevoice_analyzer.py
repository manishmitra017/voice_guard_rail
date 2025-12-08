"""
SenseVoice Analyzer Module
Unified ASR + Emotion Recognition using FunAudioLLM/SenseVoice.

Features:
- Multilingual ASR (Chinese, English, Japanese, Korean, Cantonese + 50 more)
- Speech Emotion Recognition (7 emotions)
- Audio Event Detection (laughter, applause, crying, etc.)
- 15x faster than Whisper-Large
"""

import re
import torch
import librosa
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class SenseVoiceResult:
    """Result from SenseVoice analysis."""
    text: str
    language: str
    emotion: str
    emotion_confidence: float
    audio_events: List[str]
    raw_output: str


class SenseVoiceAnalyzer:
    """
    Unified speech analysis using SenseVoice model.

    Provides:
    - Automatic Speech Recognition (50+ languages)
    - Speech Emotion Recognition (7 emotions)
    - Audio Event Detection
    """

    MODEL_ID = "FunAudioLLM/SenseVoiceSmall"
    SAMPLE_RATE = 16000

    # Emotion mapping with display info
    EMOTION_DISPLAY = {
        "HAPPY": {"label": "Happy", "emoji": "😊", "color": "#51cf66"},
        "SAD": {"label": "Sad", "emoji": "😢", "color": "#748ffc"},
        "ANGRY": {"label": "Angry", "emoji": "😠", "color": "#ff6b6b"},
        "NEUTRAL": {"label": "Neutral", "emoji": "😐", "color": "#868e96"},
        "FEARFUL": {"label": "Fearful", "emoji": "😨", "color": "#9775fa"},
        "DISGUSTED": {"label": "Disgust", "emoji": "🤢", "color": "#a9e34b"},
        "SURPRISED": {"label": "Surprised", "emoji": "😲", "color": "#ffd43b"},
        "UNKNOWN": {"label": "Unknown", "emoji": "🎭", "color": "#adb5bd"},
    }

    # Language code mapping
    LANGUAGE_MAP = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "yue": "Cantonese",
        "auto": "Auto-detect",
    }

    # Audio event patterns
    AUDIO_EVENTS = ["bgm", "applause", "laughter", "crying", "coughing", "sneezing"]

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self._model = None

    def load_model(self):
        """Load the SenseVoice model."""
        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self.MODEL_ID,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                hub="hf",
            )
            print(f"SenseVoice model loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "FunASR not installed. Install with: pip install funasr"
            )

    def _parse_sensevoice_output(self, raw_text: str) -> Tuple[str, str, List[str]]:
        """
        Parse SenseVoice rich output format.

        SenseVoice outputs text with special tags:
        <|EMOTION|> for emotions
        <|EVENT|> for audio events
        <|LANG|> for language

        Returns: (clean_text, emotion, audio_events)
        """
        emotion = "NEUTRAL"
        audio_events = []
        clean_text = raw_text

        # Extract emotion tags
        emotion_pattern = r'<\|([A-Z]+)\|>'
        matches = re.findall(emotion_pattern, raw_text)

        for match in matches:
            if match in self.EMOTION_DISPLAY:
                emotion = match
            elif match.lower() in self.AUDIO_EVENTS:
                audio_events.append(match.lower())

        # Remove all special tags from text
        clean_text = re.sub(r'<\|[^|]+\|>', '', raw_text).strip()

        return clean_text, emotion, audio_events

    def analyze(self, audio_path: str, language: str = "auto") -> Dict:
        """
        Analyze audio file for transcription, emotion, and events.

        Args:
            audio_path: Path to audio file
            language: Language code ('auto', 'zh', 'en', 'ja', 'ko', 'yue')

        Returns:
            Dictionary with transcription, emotion, and analysis results
        """
        if self._model is None:
            self.load_model()

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError:
            rich_transcription_postprocess = lambda x: x

        # Run inference
        result = self._model.generate(
            input=audio_path,
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        if not result or len(result) == 0:
            return self._empty_result()

        raw_output = result[0].get("text", "")

        # Process the rich transcription
        try:
            processed_text = rich_transcription_postprocess(raw_output)
        except:
            processed_text = raw_output

        # Parse emotion and events from output
        clean_text, emotion, audio_events = self._parse_sensevoice_output(processed_text)

        # Get display info for emotion
        emotion_info = self.EMOTION_DISPLAY.get(
            emotion,
            self.EMOTION_DISPLAY["NEUTRAL"]
        )

        # Detect language from result or default
        detected_lang = result[0].get("language", language)
        if detected_lang == "auto":
            detected_lang = "en"  # Default fallback

        # Build all probabilities (SenseVoice doesn't provide probabilities,
        # so we set detected emotion to high confidence)
        all_probs = {e: 0.05 for e in self.EMOTION_DISPLAY.keys() if e != "UNKNOWN"}
        all_probs[emotion] = 0.85  # High confidence for detected emotion

        return {
            "transcription": {
                "text": clean_text,
                "language": detected_lang,
                "language_name": self.LANGUAGE_MAP.get(detected_lang, detected_lang),
            },
            "emotion": {
                "emotion": emotion_info["label"],
                "emoji": emotion_info["emoji"],
                "color": emotion_info["color"],
                "confidence": 0.85,
                "raw_label": emotion.lower(),
                "all_probabilities": {k.lower(): v for k, v in all_probs.items()},
            },
            "audio_events": audio_events,
            "raw_output": raw_output,
        }

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            "transcription": {
                "text": "",
                "language": "unknown",
                "language_name": "Unknown",
            },
            "emotion": {
                "emotion": "Neutral",
                "emoji": "😐",
                "color": "#868e96",
                "confidence": 0.0,
                "raw_label": "neutral",
                "all_probabilities": {},
            },
            "audio_events": [],
            "raw_output": "",
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for transcription."""
        return self.LANGUAGE_MAP.copy()


# Singleton instance
_analyzer_instance: Optional[SenseVoiceAnalyzer] = None


def get_sensevoice_analyzer(use_gpu: bool = False) -> SenseVoiceAnalyzer:
    """Get or create singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SenseVoiceAnalyzer(use_gpu=use_gpu)
    return _analyzer_instance
