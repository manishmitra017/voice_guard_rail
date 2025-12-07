"""
Emotion Classifier Module
Uses Whisper-based model for 7-emotion speech recognition with 92% accuracy.

Model: firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3
Emotions: Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised
"""

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
from typing import Dict, Optional


class EmotionClassifier:
    """
    Speech emotion recognition using Whisper-based model.

    Detected emotions (7): Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised
    Accuracy: 91.99%
    """

    MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    MAX_DURATION = 30.0

    EMOTION_DISPLAY = {
        "angry": {"label": "Angry", "emoji": "😠", "color": "#ff6b6b"},
        "disgust": {"label": "Disgust", "emoji": "🤢", "color": "#a9e34b"},
        "fearful": {"label": "Fearful", "emoji": "😨", "color": "#9775fa"},
        "happy": {"label": "Happy", "emoji": "😊", "color": "#51cf66"},
        "neutral": {"label": "Neutral", "emoji": "😐", "color": "#868e96"},
        "sad": {"label": "Sad", "emoji": "😢", "color": "#748ffc"},
        "surprised": {"label": "Surprised", "emoji": "😲", "color": "#ffd43b"},
    }

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._model = None
        self._feature_extractor = None
        self._id2label = None

    def load_model(self):
        """Load the Whisper-based emotion model."""
        self._model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.MODEL_ID,
            do_normalize=True
        )
        self._id2label = self._model.config.id2label
        self._model = self._model.to(self.device)

    def _preprocess_audio(self, audio_path: str) -> dict:
        """Load and preprocess audio file."""
        audio_array, sr = librosa.load(audio_path, sr=self._feature_extractor.sampling_rate)
        max_length = int(self._feature_extractor.sampling_rate * self.MAX_DURATION)

        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=self._feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def classify(self, audio_path: str) -> Dict:
        """Classify emotion from audio file."""
        if self._model is None:
            self.load_model()

        inputs = self._preprocess_audio(audio_path)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs.cpu().numpy()[0]

        predicted_id = torch.argmax(outputs.logits, dim=-1).item()
        raw_label = self._id2label[predicted_id].lower()
        confidence = float(probs_np[predicted_id])

        display_info = self.EMOTION_DISPLAY.get(
            raw_label,
            {"label": raw_label.capitalize(), "emoji": "🎭", "color": "#868e96"}
        )

        all_probs = {}
        for i in range(len(probs_np)):
            all_probs[self._id2label[i]] = float(probs_np[i])

        return {
            "emotion": display_info["label"],
            "emoji": display_info["emoji"],
            "color": display_info["color"],
            "confidence": confidence,
            "raw_label": raw_label,
            "all_probabilities": all_probs
        }

    def is_loaded(self) -> bool:
        return self._model is not None


_classifier_instance: Optional[EmotionClassifier] = None


def get_classifier(use_gpu: bool = False) -> EmotionClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmotionClassifier(use_gpu=use_gpu)
    return _classifier_instance
