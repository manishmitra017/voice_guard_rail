"""
Speech Transcriber Module
Uses OpenAI Whisper for speech-to-text transcription.
"""

import whisper
import librosa
import numpy as np
import torch
from typing import Optional


class SpeechTranscriber:
    """
    Speech-to-text transcription using OpenAI Whisper.

    Uses 'base' model by default for balance of speed and accuracy.
    """

    WHISPER_SAMPLE_RATE = 16000

    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                       Larger = more accurate but slower.
        """
        self.model_size = model_size
        self._model = None

    def load_model(self):
        """Load the Whisper model. Call this once before transcribing."""
        self._model = whisper.load_model(self.model_size)

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio using librosa (no ffmpeg needed)."""
        audio, sr = librosa.load(audio_path, sr=self.WHISPER_SAMPLE_RATE)
        return audio.astype(np.float32)

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            Dictionary with transcription details:
            {
                "text": "The transcribed text",
                "language": "en",
            }
        """
        if self._model is None:
            self.load_model()

        # Load audio with librosa instead of ffmpeg
        audio = self._load_audio(audio_path)

        # Pad or trim to 30 seconds (Whisper requirement)
        audio = whisper.pad_or_trim(audio)

        # Create mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels).to(self._model.device)

        # Detect language
        _, probs = self._model.detect_language(mel)
        language = max(probs, key=probs.get)

        # Decode
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self._model, mel, options)

        return {
            "text": result.text.strip(),
            "language": language,
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


# Singleton instance for Streamlit
_transcriber_instance: Optional[SpeechTranscriber] = None


def get_transcriber(model_size: str = "base") -> SpeechTranscriber:
    """Get or create singleton transcriber instance."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = SpeechTranscriber(model_size=model_size)
    return _transcriber_instance
