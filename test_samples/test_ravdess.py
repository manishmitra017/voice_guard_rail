"""
Test emotion classifier against RAVDESS samples.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from emotion_classifier import EmotionClassifier

# RAVDESS emotion labels
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",  # Not in our model
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def test_samples():
    print("Loading emotion classifier...")
    classifier = EmotionClassifier(use_gpu=False)
    classifier.load_model()
    print("Model loaded!\n")

    samples_dir = os.path.join(os.path.dirname(__file__), "samples_by_emotion")

    results = []
    for filename in sorted(os.listdir(samples_dir)):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(samples_dir, filename)
        expected = filename.split("_")[1].replace(".wav", "")

        print(f"Testing: {filename}")
        print(f"  Expected: {expected}")

        result = classifier.classify(filepath)
        predicted = result["emotion"].lower()
        confidence = result["confidence"]

        match = "✅" if expected == predicted else "❌"
        # Calm maps to neutral in our model
        if expected == "calm" and predicted == "neutral":
            match = "✅ (calm→neutral)"

        print(f"  Predicted: {predicted} ({confidence:.1%}) {match}")
        print(f"  All scores: {', '.join(f'{k}:{v:.1%}' for k, v in sorted(result['all_probabilities'].items(), key=lambda x: -x[1])[:3])}")
        print()

        results.append({
            "file": filename,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "correct": expected == predicted or (expected == "calm" and predicted == "neutral")
        })

    # Summary
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print("=" * 50)
    print(f"Results: {correct}/{total} correct ({correct/total:.1%})")
    print("=" * 50)

if __name__ == "__main__":
    test_samples()
