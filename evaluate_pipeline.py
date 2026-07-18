from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from loop4 import run_interaction_pipeline


DEFAULT_GROUND_TRUTH = {
    "happy_face.jpg": "happy",
    "sad_face.jpg": "sad",
    "neutral_face.jpg": "neutral",
    "angry_face.jpg": "angry",
    "gloomy_face.jpg": "gloomy",
}

EMOTION_KEYWORDS = {
    "happy": ("happy", "smile", "smiling", "joy", "cheerful", "positive"),
    "sad": ("sad", "unhappy", "downcast", "cry", "tear", "upset"),
    "angry": ("angry", "anger", "mad", "furious", "irritated", "annoyed"),
    "neutral": ("neutral", "calm", "relaxed", "expressionless", "blank"),
    "gloomy": ("gloomy", "somber", "sombre", "tired", "depressed", "down"),
}

EMPATHY_PHRASES = (
    "무슨 일 있으세요",
    "괜찮으세요",
    "힘든 일",
    "속상",
    "도와드릴까요",
    "말씀해 주세요",
)

NEGATIVE_EMOTIONS = {"sad", "angry", "gloomy"}


def predict_emotion(description: str) -> str:
    lowered = (description or "").lower()
    scores = {
        emotion: sum(1 for keyword in keywords if keyword in lowered)
        for emotion, keywords in EMOTION_KEYWORDS.items()
    }
    predicted, score = max(scores.items(), key=lambda item: item[1])
    return predicted if score > 0 else "unknown"


def score_visual_accuracy(expected: str, predicted: str) -> float:
    if predicted == expected:
        return 1.0
    if {expected, predicted} <= {"sad", "gloomy"}:
        return 0.75
    if predicted == "unknown":
        return 0.0
    return 0.25


def score_exaone_quality(expected_emotion: str, response: str) -> tuple[float, list[str]]:
    issues: list[str] = []
    text = response or ""
    has_korean = bool(re.search(r"[가-힣]", text))
    has_empathy = any(phrase in text for phrase in EMPATHY_PHRASES)
    too_long = len(text) > 180
    repetitive = len(re.findall(r"(.{4,}?)\1", text)) > 0
    character_break = any(token in text.lower() for token in ("as an ai", "language model", "analysis", "markdown"))

    score = 1.0
    if not has_korean:
        score -= 0.35
        issues.append("Korean text missing")
    if expected_emotion in NEGATIVE_EMOTIONS and not has_empathy:
        score -= 0.35
        issues.append("negative emotion did not trigger empathetic check-in")
    if expected_emotion not in NEGATIVE_EMOTIONS and has_empathy:
        score -= 0.15
        issues.append("unnecessary concern for non-negative emotion")
    if too_long:
        score -= 0.10
        issues.append("response too long for robot speech")
    if repetitive:
        score -= 0.10
        issues.append("repetitive output")
    if character_break:
        score -= 0.20
        issues.append("broke Jarvis character")
    return max(0.0, score), issues


def score_latency(timings: dict[str, float]) -> float:
    total = float(timings.get("total_sec", 999.0))
    moondream = float(timings.get("moondream_sec", 999.0))
    exaone = float(timings.get("exaone_sec", 999.0))
    tts = float(timings.get("tts_sec", 999.0))

    score = 1.0
    if moondream > 15.0:
        score -= min(0.30, (moondream - 15.0) / 60.0)
    if exaone > 10.0:
        score -= min(0.25, (exaone - 10.0) / 40.0)
    if tts > 5.0:
        score -= min(0.15, (tts - 5.0) / 30.0)
    if total > 30.0:
        score -= min(0.30, (total - 30.0) / 90.0)
    return max(0.0, score)


def evaluate_case(image_path: Path, expected_emotion: str, output_dir: Path, play_audio: bool) -> dict[str, Any]:
    result = run_interaction_pipeline(
        image_path,
        output_path=str(output_dir / f"{image_path.stem}_response.mp3"),
        play_audio=play_audio,
    )
    predicted_emotion = predict_emotion(result["moondream_raw"])
    visual_score = score_visual_accuracy(expected_emotion, predicted_emotion)
    exaone_score, issues = score_exaone_quality(expected_emotion, result["exaone_text"])
    latency_score = score_latency(result["timings"])
    total_score = (visual_score * 40.0) + (exaone_score * 35.0) + (latency_score * 25.0)

    return {
        "image": image_path.name,
        "expected": expected_emotion,
        "predicted": predicted_emotion,
        "success": result["success"],
        "moondream_raw": result["moondream_raw"],
        "exaone_text": result["exaone_text"],
        "tts_status": result["tts_status"],
        "timings": result["timings"],
        "visual_score": round(visual_score * 40.0, 2),
        "exaone_score": round(exaone_score * 35.0, 2),
        "latency_score": round(latency_score * 25.0, 2),
        "total_score": round(total_score, 2),
        "issues": issues if result["success"] else [result["error"]],
    }


def print_confusion_matrix(rows: list[dict[str, Any]]) -> None:
    labels = ["happy", "sad", "angry", "neutral", "gloomy", "unknown"]
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1

    print("\n[Visual Perception Matrix] ground_truth x predicted")
    print("truth\\pred".ljust(14) + " ".join(label.rjust(8) for label in labels))
    for truth in labels[:-1]:
        values = " ".join(str(matrix[truth][pred]).rjust(8) for pred in labels)
        print(truth.ljust(14) + values)


def print_scorecard(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No evaluation rows produced.")
        return

    print("\n===== Performance Scorecard =====")
    print(f"Visual Perception Accuracy: {mean(row['visual_score'] for row in rows):.2f} / 40")
    print(f"Contextual Inference Quality: {mean(row['exaone_score'] for row in rows):.2f} / 35")
    print(f"End-to-End Latency & Performance: {mean(row['latency_score'] for row in rows):.2f} / 25")
    print(f"Overall Score: {mean(row['total_score'] for row in rows):.2f} / 100")
    print(f"Average Total Latency: {mean(row['timings']['total_sec'] for row in rows):.2f}s")

    print("\n[Case Details]")
    for row in rows:
        print(
            f"- {row['image']}: expected={row['expected']} predicted={row['predicted']} "
            f"score={row['total_score']}/100 total_latency={row['timings']['total_sec']}s"
        )
        print(f"  moondream={row['moondream_raw']}")
        print(f"  exaone={row['exaone_text']}")
        if row["issues"]:
            print(f"  issues={'; '.join(row['issues'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective evaluator for Moondream -> EXAONE -> TTS pipeline")
    parser.add_argument("--image-dir", default="test_images", help="Directory containing labeled expression images")
    parser.add_argument("--output-dir", default="eval_pipeline", help="Directory for generated TTS MP3 files")
    parser.add_argument("--play", action="store_true", help="Play generated TTS audio during evaluation")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (image_dir / filename, emotion)
        for filename, emotion in DEFAULT_GROUND_TRUTH.items()
        if (image_dir / filename).exists()
    ]
    if not cases:
        expected = ", ".join(DEFAULT_GROUND_TRUTH)
        raise SystemExit(f"평가 이미지가 없습니다. {image_dir}/ 아래에 다음 예시 파일을 넣어주세요: {expected}")

    rows = [evaluate_case(image_path, emotion, output_dir, play_audio=args.play) for image_path, emotion in cases]
    print_confusion_matrix(rows)
    print_scorecard(rows)


if __name__ == "__main__":
    main()