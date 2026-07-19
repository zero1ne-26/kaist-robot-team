from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.loop4 import EMOTION_RULES, run_interaction_pipeline, sanitize_tts_text, warm_up_models


EMOTIONS = ("happy", "sad", "angry", "surprised", "neutral")
SAMPLES_PER_EMOTION = 4
DEFAULT_DATASET_DIR = Path("test_images/batch_mock_20")
DEFAULT_OUTPUT_DIR = Path("eval_pipeline/batch_20")

MOONDREAM_KEYWORDS = {
    "happy": ("happy", "smile", "smiling", "joy", "cheerful", "pleased"),
    "sad": ("sad", "gloomy", "upset", "unhappy", "down", "frown"),
    "angry": ("angry", "mad", "annoyed", "irritated", "furious", "stern"),
    "surprised": ("surprised", "surprise", "shocked", "astonished", "wide eyes", "open mouth"),
    "neutral": ("neutral", "calm", "plain", "expressionless", "relaxed", "normal"),
}

BAD_MOOD_PHRASES = ("무슨 일 있으세요", "괜찮으세요", "힘든", "속상", "도와드릴까요", "천천히")
HAPPY_PHRASES = ("좋은 일", "기뻐", "공유", "함께", "좋아 보")
ANGRY_PHRASES = ("화가", "속상", "차분", "도와", "괜찮")
SURPRISED_PHRASES = ("놀라", "무슨 일", "괜찮", "무슨 일이")
NEUTRAL_PHRASES = ("안녕하세요", "필요", "도와", "좋은 하루")
FORBIDDEN_HAPPY_CONCERN = ("무슨 일 있으세요", "괜찮으세요", "힘든 일")


def draw_mock_face(emotion: str, variant: int, output_path: Path, label_overlay: bool = True) -> None:
    """Create a simple cartoon face for deterministic local pipeline testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg_colors = {
        "happy": (218, 245, 235),
        "sad": (215, 225, 240),
        "angry": (225, 215, 210),
        "surprised": (230, 235, 248),
        "neutral": (232, 232, 232),
    }
    image = np.full((480, 640, 3), bg_colors[emotion], dtype=np.uint8)
    center = (320 + (variant - 2) * 5, 225)
    face_color = (190, 210, 232)
    line_color = (35, 45, 60)

    cv2.circle(image, center, 132, face_color, -1)
    left_eye = (center[0] - 50, center[1] - 30)
    right_eye = (center[0] + 50, center[1] - 30)

    if emotion == "happy":
        cv2.ellipse(image, left_eye, (22, 12), 0, 0, 180, line_color, 5)
        cv2.ellipse(image, right_eye, (22, 12), 0, 0, 180, line_color, 5)
        cv2.ellipse(image, (center[0], center[1] + 45), (58, 40), 0, 15, 165, line_color, 6)
    elif emotion == "sad":
        cv2.circle(image, left_eye, 16, line_color, -1)
        cv2.circle(image, right_eye, 16, line_color, -1)
        cv2.line(image, (center[0] - 82, center[1] - 65), (center[0] - 35, center[1] - 52), line_color, 5)
        cv2.line(image, (center[0] + 82, center[1] - 65), (center[0] + 35, center[1] - 52), line_color, 5)
        cv2.ellipse(image, (center[0], center[1] + 72), (56, 34), 0, 200, 340, line_color, 6)
    elif emotion == "angry":
        cv2.circle(image, left_eye, 16, line_color, -1)
        cv2.circle(image, right_eye, 16, line_color, -1)
        cv2.line(image, (center[0] - 82, center[1] - 78), (center[0] - 25, center[1] - 50), line_color, 6)
        cv2.line(image, (center[0] + 82, center[1] - 78), (center[0] + 25, center[1] - 50), line_color, 6)
        cv2.line(image, (center[0] - 48, center[1] + 65), (center[0] + 48, center[1] + 65), line_color, 6)
    elif emotion == "surprised":
        cv2.circle(image, left_eye, 23, line_color, 5)
        cv2.circle(image, right_eye, 23, line_color, 5)
        cv2.circle(image, (center[0], center[1] + 58), 34, line_color, 6)
        cv2.line(image, (center[0] - 82, center[1] - 72), (center[0] - 35, center[1] - 82), line_color, 5)
        cv2.line(image, (center[0] + 82, center[1] - 72), (center[0] + 35, center[1] - 82), line_color, 5)
    else:
        cv2.circle(image, left_eye, 16, line_color, -1)
        cv2.circle(image, right_eye, 16, line_color, -1)
        cv2.line(image, (center[0] - 46, center[1] + 58), (center[0] + 46, center[1] + 58), line_color, 5)

    if label_overlay:
        cv2.putText(image, emotion, (245, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.1, line_color, 3, cv2.LINE_AA)

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to write mock image: {output_path}")


def create_mock_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR, label_overlay: bool = True) -> list[dict[str, Any]]:
    """Generate 20 mock images: five emotions, four variants each."""
    records: list[dict[str, Any]] = []
    for emotion in EMOTIONS:
        for variant in range(1, SAMPLES_PER_EMOTION + 1):
            path = dataset_dir / emotion / f"{emotion}_{variant:02d}.jpg"
            draw_mock_face(emotion, variant, path, label_overlay=label_overlay)
            records.append({"image_path": path, "expected_emotion": emotion, "variant": variant})
    return records


def predict_emotion_from_moondream(text: str) -> str:
    lowered = (text or "").lower()
    scores = {
        emotion: sum(1 for keyword in keywords if keyword in lowered)
        for emotion, keywords in MOONDREAM_KEYWORDS.items()
    }
    emotion, score = max(scores.items(), key=lambda item: item[1])
    return emotion if score > 0 else "unknown"


def contains_special_tts_risk(text: str) -> bool:
    return sanitize_tts_text(text) != (text or "").strip()


def validate_reasoning(expected: str, response: str) -> tuple[bool, str]:
    if not response.strip():
        return False, "empty EXAONE response"
    if not re.search(r"[가-힣]", response):
        return False, "Korean response missing"
    if expected == "happy":
        if any(phrase in response for phrase in FORBIDDEN_HAPPY_CONCERN):
            return False, "happy response incorrectly asks if something is wrong"
        if not any(phrase in response for phrase in HAPPY_PHRASES):
            return False, "happy response does not share joy"
    elif expected == "sad":
        if not any(phrase in response for phrase in BAD_MOOD_PHRASES):
            return False, "sad response lacks empathetic check-in"
    elif expected == "angry":
        if not any(phrase in response for phrase in ANGRY_PHRASES):
            return False, "angry response lacks calming/helpful tone"
    elif expected == "surprised":
        if not any(phrase in response for phrase in SURPRISED_PHRASES):
            return False, "surprised response lacks gentle inquiry"
    elif expected == "neutral":
        if any(phrase in response for phrase in FORBIDDEN_HAPPY_CONCERN):
            return False, "neutral response overreacts with concern"
        if not any(phrase in response for phrase in NEUTRAL_PHRASES):
            return False, "neutral response lacks natural greeting/help offer"
    return True, "ok"


def run_case(record: dict[str, Any], output_dir: Path, play_audio: bool) -> dict[str, Any]:
    image_path = Path(record["image_path"])
    expected = str(record["expected_emotion"])
    started = time.perf_counter()
    result = run_interaction_pipeline(
        image_path,
        user_text=f"사용자가 카메라 앞에 있습니다. 기대 감정 라벨은 평가용으로 {expected}입니다.",
        output_path=str(output_dir / f"{image_path.stem}_response.mp3"),
        play_audio=play_audio,
        warmup=False,
    )
    predicted = predict_emotion_from_moondream(result["moondream_raw"])
    reasoning_ok, reasoning_issue = validate_reasoning(expected, result["exaone_text"])
    tts_risk = contains_special_tts_risk(result["exaone_text"])

    return {
        "image": str(image_path),
        "expected": expected,
        "predicted": predicted,
        "vision_ok": predicted == expected,
        "reasoning_ok": reasoning_ok,
        "reasoning_issue": reasoning_issue,
        "tts_ok": result["tts_status"].get("ok", False) and not tts_risk,
        "tts_special_char_risk": tts_risk,
        "moondream_raw": result["moondream_raw"],
        "exaone_text": result["exaone_text"],
        "error": result["error"],
        "moondream_sec": result["timings"]["moondream_sec"],
        "exaone_sec": result["timings"]["exaone_sec"],
        "tts_sec": result["timings"]["tts_sec"],
        "total_sec": result["timings"]["total_sec"],
        "wall_sec": round(time.perf_counter() - started, 3),
    }


def confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    return matrix


def fail_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return 1.0 - (sum(1 for row in rows if row[key]) / len(rows))


def markdown_report(rows: list[dict[str, Any]], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg_moondream = mean(row["moondream_sec"] for row in rows) if rows else 0.0
    avg_exaone = mean(row["exaone_sec"] for row in rows) if rows else 0.0
    avg_tts = mean(row["tts_sec"] for row in rows) if rows else 0.0
    avg_total = mean(row["total_sec"] for row in rows) if rows else 0.0
    bottleneck = max(
        (("Moondream", avg_moondream), ("EXAONE", avg_exaone), ("TTS", avg_tts)),
        key=lambda item: item[1],
    )[0]
    labels = list(EMOTIONS) + ["unknown"]
    matrix = confusion_matrix(rows)

    lines = [
        "# Moondream -> EXAONE -> TTS Batch Evaluation",
        "",
        "## Summary",
        f"- Samples: {len(rows)}",
        f"- Vision fail-rate: {fail_rate(rows, 'vision_ok') * 100:.1f}%",
        f"- Reasoning fail-rate: {fail_rate(rows, 'reasoning_ok') * 100:.1f}%",
        f"- TTS compatibility fail-rate: {fail_rate(rows, 'tts_ok') * 100:.1f}%",
        f"- Avg Moondream latency: {avg_moondream:.2f}s",
        f"- Avg EXAONE latency: {avg_exaone:.2f}s",
        f"- Avg TTS latency: {avg_tts:.2f}s",
        f"- Avg total latency: {avg_total:.2f}s",
        f"- Primary bottleneck: {bottleneck}",
        "",
        "## Emotion Prompt Rules",
    ]
    for emotion, rule in EMOTION_RULES.items():
        lines.append(f"- {emotion}: {rule}")

    lines.extend(["", "## Vision Confusion Matrix", "", "| truth \\ predicted | " + " | ".join(labels) + " |"])
    lines.append("|---|" + "|".join("---" for _ in labels) + "|")
    for truth in EMOTIONS:
        lines.append("| " + truth + " | " + " | ".join(str(matrix[truth][pred]) for pred in labels) + " |")

    lines.extend([
        "",
        "## Per-Sample Matrix",
        "",
        "| image | expected | predicted | vision | reasoning | tts | moondream_s | exaone_s | tts_s | total_s | issue |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        issue = row["reasoning_issue"] if row["reasoning_issue"] != "ok" else (row["error"] or "")
        lines.append(
            f"| {Path(row['image']).name} | {row['expected']} | {row['predicted']} | "
            f"{row['vision_ok']} | {row['reasoning_ok']} | {row['tts_ok']} | "
            f"{row['moondream_sec']:.2f} | {row['exaone_sec']:.2f} | {row['tts_sec']:.2f} | {row['total_sec']:.2f} | {issue} |"
        )

    report = "\n".join(lines) + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_terminal_scorecard(rows: list[dict[str, Any]], report_path: Path) -> None:
    if not rows:
        print("No rows evaluated.")
        return
    avg_moondream = mean(row["moondream_sec"] for row in rows)
    avg_exaone = mean(row["exaone_sec"] for row in rows)
    avg_tts = mean(row["tts_sec"] for row in rows)
    avg_total = mean(row["total_sec"] for row in rows)
    print("\n===== Batch Performance Scorecard =====")
    print(f"samples={len(rows)}")
    print(f"vision_accuracy={(1 - fail_rate(rows, 'vision_ok')) * 100:.1f}%")
    print(f"reasoning_pass_rate={(1 - fail_rate(rows, 'reasoning_ok')) * 100:.1f}%")
    print(f"tts_compatibility={(1 - fail_rate(rows, 'tts_ok')) * 100:.1f}%")
    print(f"avg_moondream_sec={avg_moondream:.2f}")
    print(f"avg_exaone_sec={avg_exaone:.2f}")
    print(f"avg_tts_sec={avg_tts:.2f}")
    print(f"avg_total_sec={avg_total:.2f}")
    print(f"report={report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="20-sample evaluator and optimizer for Moondream -> EXAONE -> TTS")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Where to create/read the 20 mock images")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to write audio, CSV, and Markdown reports")
    parser.add_argument("--no-label-overlay", action="store_true", help="Do not draw emotion words on mock images")
    parser.add_argument("--skip-run", action="store_true", help="Only generate the mock dataset; do not call Ollama/TTS")
    parser.add_argument("--play", action="store_true", help="Play generated TTS audio for every sample")
    parser.add_argument("--no-warmup", action="store_true", help="Skip Ollama model warm-up")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = create_mock_dataset(dataset_dir, label_overlay=not args.no_label_overlay)
    print(f"[DATASET] generated={len(records)} dir={dataset_dir}")
    if args.skip_run:
        return

    if not args.no_warmup:
        print("[WARMUP] keeping Ollama models loaded...")
        warm_up_models()

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        print(f"[{index:02d}/{len(records)}] {record['expected_emotion']} {record['image_path']}")
        row = run_case(record, output_dir, play_audio=args.play)
        rows.append(row)
        print(
            f"    vision={row['predicted']} ok={row['vision_ok']} "
            f"reasoning_ok={row['reasoning_ok']} tts_ok={row['tts_ok']} total={row['total_sec']:.2f}s"
        )

    csv_path = output_dir / "batch_results.csv"
    report_path = output_dir / "batch_report.md"
    write_csv(rows, csv_path)
    markdown_report(rows, report_path)
    print_terminal_scorecard(rows, report_path)


if __name__ == "__main__":
    main()