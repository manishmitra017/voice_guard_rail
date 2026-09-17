#!/usr/bin/env python3
"""
Download every model the app needs into models/ so it can run without
HuggingFace access. Run this on a machine that CAN reach huggingface.co,
then commit models/ (Git LFS) and clone the repo at the office.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --only sensevoice-small fsmn-vad
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_paths import MODELS, MODELS_DIR, whisper_download_root  # noqa: E402

# Files that are never needed at inference time and only bloat the repo.
# requirements.txt is excluded on purpose: funasr pip-installs it at load time
# (gradio, modelscope), which is slow and pointless when deps come from pyproject.toml.
IGNORE = [
    "*.msgpack", "*.h5", "*.onnx", "*.tflite", "*.ot", "*.md", ".gitattributes",
    "requirements.txt", "image/*", "fig/*", "example/*", "demo*",
]


def download_hf(name: str, repo_id: str):
    from huggingface_hub import snapshot_download

    target = MODELS_DIR / name
    print(f"\n==> {repo_id}\n    -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        ignore_patterns=IGNORE,
    )


# GitHub rejects any single LFS object over 2GiB, so large checkpoints are
# re-saved as sharded safetensors. from_pretrained() reassembles shards
# transparently via the generated index file.
MAX_SHARD_SIZE = "1GB"
SHARDED = {
    "whisper-emotion": "AutoModelForAudioClassification",
    "nllb-200-distilled-600M": "AutoModelForSeq2SeqLM",
}


def reshard(name: str):
    import transformers

    target = MODELS_DIR / name
    oversized = [f for f in target.glob("*") if f.is_file() and f.stat().st_size > 2 * 1024**3]
    if not oversized:
        return

    print(f"    resharding {name} ({', '.join(f.name for f in oversized)})")
    cls = getattr(transformers, SHARDED[name])
    model = cls.from_pretrained(target)
    model.save_pretrained(target, max_shard_size=MAX_SHARD_SIZE, safe_serialization=True)

    for f in oversized:
        f.unlink()


def download_whisper():
    import whisper

    root = whisper_download_root()
    print(f"\n==> openai-whisper 'base'\n    -> {root}")
    whisper.load_model("base", download_root=root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="subset of model names to fetch")
    args = parser.parse_args()

    names = args.only or list(MODELS) + ["whisper"]

    for name in names:
        if name == "whisper":
            download_whisper()
        elif name in MODELS:
            download_hf(name, MODELS[name])
            if name in SHARDED:
                reshard(name)
        else:
            sys.exit(f"Unknown model '{name}'. Choose from: {list(MODELS) + ['whisper']}")

    print(f"\nDone. Models are in {MODELS_DIR}")
    print("Next: git add models && git commit && git push  (Git LFS handles the weights)")


if __name__ == "__main__":
    main()
