"""
Voice Emotion Recognition App
Streamlit web application for real-time speech emotion detection.

Emotion Model: firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3
Emotions: Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprised (7 emotions, 92% accuracy)
Transcription: OpenAI Whisper
"""

import warnings
import os
import sys

# Suppress SpeechBrain lazy import warnings
warnings.filterwarnings("ignore", message=".*was deprecated.*")
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", message=".*FP16.*")

# Suppress logging noise before imports
import logging
logging.getLogger("speechbrain").setLevel(logging.WARNING)

import streamlit as st

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_recorder import AudioRecorder
from emotion_classifier import get_classifier
from speech_transcriber import get_transcriber

# Page configuration
st.set_page_config(
    page_title="Voice Emotion Detector",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for better styling
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
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 20px;
    }
    .recording-indicator {
        color: #ff4b4b;
        font-size: 24px;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎙️ Voice Emotion Detector")
st.markdown("Detect emotions from your voice using AI")
st.markdown("---")


# Initialize session state
if 'recorder' not in st.session_state:
    st.session_state.recorder = AudioRecorder()
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


# Load models on first run
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


# Recording controls
col1, col2 = st.columns(2)

with col1:
    if st.button(
        "🎙️ Start Recording" if not st.session_state.is_recording else "⏹️ Stop Recording",
        type="primary" if not st.session_state.is_recording else "secondary",
        use_container_width=True
    ):
        if not st.session_state.is_recording:
            # Start recording
            st.session_state.recorder.start_recording()
            st.session_state.is_recording = True
            st.session_state.result = None
            st.session_state.transcript = None
            st.rerun()
        else:
            # Stop recording and analyze
            st.session_state.is_recording = False
            audio_path = st.session_state.recorder.stop_recording()

            if audio_path:
                with st.spinner("🔍 Analyzing speech and emotion..."):
                    try:
                        # Transcribe speech
                        st.session_state.transcript = transcriber.transcribe(audio_path)

                        # Classify emotion
                        st.session_state.result = classifier.classify(audio_path)

                        # Clean up temp file
                        os.unlink(audio_path)
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        st.session_state.result = None
                        st.session_state.transcript = None
            else:
                st.warning("No audio recorded. Please try again.")
            st.rerun()

with col2:
    if st.session_state.is_recording:
        st.markdown(
            '<p class="recording-indicator">🔴 Recording...</p>',
            unsafe_allow_html=True
        )
        st.caption("Speak now, then click Stop when done")


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

    # Get color from result (new model provides it)
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
