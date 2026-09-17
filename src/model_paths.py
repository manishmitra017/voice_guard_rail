"""
Model Path Resolution
Prefers models vendored in-repo so the app runs without HuggingFace access.

Set VG_MODELS_DIR to point elsewhere (e.g. a shared network drive).
Set VG_OFFLINE=1 to fail loudly instead of silently falling back to the hub.
"""

import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("VG_MODELS_DIR", Path(__file__).parent.parent / "models"))

OFFLINE = os.environ.get("VG_OFFLINE", "").lower() in ("1", "true", "yes")

# Local directory name -> HuggingFace repo id
MODELS = {
    "sensevoice-small": "FunAudioLLM/SenseVoiceSmall",
    "fsmn-vad": "funasr/fsmn-vad",
    "whisper-emotion": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    "nllb-200-distilled-600M": "facebook/nllb-200-distilled-600M",
}


def resolve(name: str) -> str:
    """
    Return a local path for a vendored model, or its HuggingFace id as fallback.

    Args:
        name: key from MODELS (also the directory name under MODELS_DIR).
    """
    path = MODELS_DIR / name
    if path.is_dir() and any(path.iterdir()):
        return str(path)

    if OFFLINE:
        raise FileNotFoundError(
            f"VG_OFFLINE is set but '{name}' is not vendored at {path}.\n"
            f"Run 'python scripts/download_models.py' on a machine with HuggingFace access."
        )

    return MODELS[name]


def whisper_download_root() -> str:
    """Directory openai-whisper loads its .pt checkpoints from."""
    root = MODELS_DIR / "whisper"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)
