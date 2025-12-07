"""
Voice Emotion Recognition App (Cloud Version)
Streamlit web application for real-time speech emotion detection.
Uses browser-based audio recording for cloud deployment compatibility.

Emotion Model: firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3
Emotions: Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised (7 emotions, 92% accuracy)
Transcription: OpenAI Whisper
"""

import warnings
import os
import sys
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore", message=".*was deprecated.*")
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", message=".*FP16.*")

import logging
logging.getLogger("speechbrain").setLevel(logging.WARNING)

import streamlit as st
from streamlit_mic_recorder import mic_recorder

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from emotion_classifier import get_classifier
from speech_transcriber import get_transcriber

# Page configuration
st.set_page_config(
    page_title="Voice Emotion Detector",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .emotion-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .emotion-emoji {
        font-size: 80px;
    }
    .emotion-label {
        font-size: 36px;
        font-weight: bold;
        margin-top: 10px;
    }
    .confidence-text {
        font-size: 18px;
        color: #666;
    }
    .transcript-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #4dabf7;
        margin: 20px 0;
    }
    .transcript-label {
        font-size: 14px;
        color: #868e96;
        margin-bottom: 8px;
    }
    .transcript-text {
        font-size: 18px;
        color: #212529;
        font-style: italic;
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e7f5ff;
        border: 1px solid #74c0fc;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎙️ Voice Emotion Detector")
st.markdown("Detect emotions from your voice using AI")
st.markdown("---")

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


# Load models
@st.cache_resource
def load_emotion_model():
    """Load the emotion classification model (cached)."""
    classifier = get_classifier(use_gpu=False)
    classifier.load_model()
    return classifier


@st.cache_resource
def load_whisper_model():
    """Load the Whisper transcription model (cached)."""
    transcriber = get_transcriber(model_size="base")
    transcriber.load_model()
    return transcriber


# Model loading section
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading AI models... This may take a minute on first run."):
        try:
            classifier = load_emotion_model()
            transcriber = load_whisper_model()
            st.session_state.model_loaded = True
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load models: {e}")
            st.stop()
else:
    classifier = load_emotion_model()
    transcriber = load_whisper_model()

# Instructions
st.markdown("""
<div class="info-box">
    <strong>📋 How to use:</strong><br>
    1. Click the microphone button below to start recording<br>
    2. Speak with emotion (happy, angry, sad, etc.)<br>
    3. Click again to stop - analysis will start automatically
</div>
""", unsafe_allow_html=True)

# Browser-based audio recorder
audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=True,
    format="wav",
    callback=None,
    key="voice_recorder"
)

# Process audio when recorded
if audio:
    audio_bytes = audio["bytes"]

    if audio_bytes:
        # Show audio player
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("🔍 Analyzing speech and emotion..."):
            try:
                # Save audio to temp file for processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                # Transcribe speech
                st.session_state.transcript = transcriber.transcribe(tmp_path)

                # Classify emotion
                st.session_state.result = classifier.classify(tmp_path)

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.session_state.result = None
                st.session_state.transcript = None

# Display results
if st.session_state.result:
    result = st.session_state.result
    transcript = st.session_state.transcript

    st.markdown("---")

    # Show transcription first
    if transcript and transcript.get("text"):
        st.subheader("📝 What You Said")
        st.markdown(f"""
        <div class="transcript-box">
            <div class="transcript-label">Transcribed Text:</div>
            <div class="transcript-text">"{transcript['text']}"</div>
        </div>
        """, unsafe_allow_html=True)

    # Show emotion
    st.subheader("🎭 Detected Emotion")
    bg_color = result.get("color", "#868e96")

    st.markdown(f"""
    <div class="emotion-box" style="background-color: {bg_color}20; border: 2px solid {bg_color};">
        <div class="emotion-emoji">{result["emoji"]}</div>
        <div class="emotion-label" style="color: {bg_color};">{result["emotion"]}</div>
        <div class="confidence-text">Confidence: {result["confidence"]:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Show all emotion probabilities
    if "all_probabilities" in result:
        with st.expander("📊 All Emotion Scores"):
            probs = result["all_probabilities"]
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_probs:
                st.progress(prob, text=f"{emotion}: {prob:.1%}")

    # Show raw details in expander
    with st.expander("🔬 Technical Details"):
        st.json({
            "transcript": transcript.get("text", "") if transcript else "",
            "language_detected": transcript.get("language", "unknown") if transcript else "unknown",
            "emotion_raw_label": result["raw_label"],
            "emotion_confidence": result["confidence"],
            "emotion_model": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
            "transcription_model": "openai/whisper-base"
        })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>Powered by <a href="https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3">Whisper Emotion Model</a> &
    <a href="https://github.com/openai/whisper">OpenAI Whisper</a></p>
    <p>7 Emotions: Angry 😠 | Disgust 🤢 | Fearful 😨 | Happy 😊 | Neutral 😐 | Sad 😢 | Surprised 😲</p>
    <p style="font-size: 12px; color: #adb5bd;">92% accuracy on emotion detection</p>
</div>
""", unsafe_allow_html=True)
