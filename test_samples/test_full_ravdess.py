"""
Full test of emotion classifier against all RAVDESS actors.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from emotion_classifier import EmotionClassifier
from collections import defaultdict

# RAVDESS emotion codes
RAVDESS_TO_LABEL = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def test_all_actors():
    print("Loading emotion classifier...")
    classifier = EmotionClassifier(use_gpu=False)
    classifier.load_model()
    print("Model loaded!\n")

    base_dir = os.path.dirname(__file__)

    # Track results per emotion
    emotion_results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": defaultdict(int)})

    # Test first 4 actors (2 male, 2 female) with high intensity samples
    actors = ["Actor_01", "Actor_02", "Actor_03", "Actor_04"]

    for actor in actors:
        actor_dir = os.path.join(base_dir, actor)
        if not os.path.exists(actor_dir):
            continue

        print(f"\n{'='*50}")
        print(f"Testing {actor}")
        print('='*50)

        for filename in sorted(os.listdir(actor_dir)):
            if not filename.endswith(".wav"):
                continue

            # Parse filename: modality-channel-emotion-intensity-statement-repetition-actor
            parts = filename.replace(".wav", "").split("-")
            emotion_code = parts[2]
            intensity = parts[3]

            # Only test high intensity (02) for clearer emotions
            if intensity != "02":
                continue

            expected = RAVDESS_TO_LABEL.get(emotion_code, "unknown")
            if expected == "unknown":
                continue

            filepath = os.path.join(actor_dir, filename)
            result = classifier.classify(filepath)
            predicted = result["emotion"].lower()
            confidence = result["confidence"]

            # Check correctness (calm maps to neutral)
            correct = expected == predicted or (expected == "calm" and predicted == "neutral")

            emotion_results[expected]["total"] += 1
            emotion_results[expected]["predictions"][predicted] += 1
            if correct:
                emotion_results[expected]["correct"] += 1

            match = "✅" if correct else "❌"
            if expected == "calm" and predicted == "neutral":
                match = "✅"
            print(f"  {expected:10} → {predicted:10} ({confidence:.0%}) {match}")

    # Summary
    print("\n" + "="*60)
    print("EMOTION-WISE ACCURACY")
    print("="*60)

    total_correct = 0
    total_samples = 0

    for emotion in sorted(emotion_results.keys()):
        data = emotion_results[emotion]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        total_correct += data["correct"]
        total_samples += data["total"]

        # Top prediction
        top_pred = max(data["predictions"].items(), key=lambda x: x[1])[0]
        print(f"{emotion:12}: {data['correct']:2}/{data['total']:2} ({acc:5.1f}%) | Most predicted: {top_pred}")

    print("="*60)
    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"OVERALL: {total_correct}/{total_samples} ({overall:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    test_all_actors()
