from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from interactive_scenario import call_exaone_contextual_response, EXAONE_HRI_SYSTEM_PROMPT, speak_tts
from mediapipe_emotion import create_face_landmarker, emotion_scores_from_result


TEST_CASES = [
    {"image": "sad_face.jpg", "expected": {"sad"}, "user_text": "오늘 표정이 안 좋아 보여"},
    {"image": "mad.jpg", "expected": {"angry"}, "user_text": "지금 너무 화가 나"},
    {"image": "happt.jpg", "expected": {"happy"}, "user_text": "오늘 기분이 정말 좋아"},
    {"image": "anxi.jpg", "expected": {"fear"}, "user_text": "괜히 불안하고 긴장돼"},
]


def detect_image_emotion(landmarker, image_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    scores, _face_landmarks = emotion_scores_from_result(result)
    dominant = max(scores, key=scores.get)
    return {
        "dominant": dominant,
        "dominant_score": scores[dominant],
        "scores": scores,
        "latency_sec": round(time.perf_counter() - started, 3),
    }


def run_case(landmarker, case: dict[str, Any], output_dir: Path, play_audio: bool) -> dict[str, Any]:
    detection = detect_image_emotion(landmarker, case["image"])
    expected = set(case["expected"])
    vision_ok = detection["dominant"] in expected
    emotion = detection["dominant"]
    history = [
        {"role": "system", "content": EXAONE_HRI_SYSTEM_PROMPT},
        {"role": "assistant", "content": "표정이 평소와 달라 보여서 먼저 말을 걸었어요."},
        {"role": "user", "content": case["user_text"]},
    ]

    exaone_started = time.perf_counter()
    response = call_exaone_contextual_response(
        history,
        vision_description=f"MediaPipe blendshape dominant emotion: {emotion}, score={detection['dominant_score']:.2f}",
        emotion=emotion,
    )
    exaone_sec = round(time.perf_counter() - exaone_started, 3)

    tts_started = time.perf_counter()
    tts_status = speak_tts(
        response,
        output_path=str(output_dir / f"{Path(case['image']).stem}_mediapipe_response.mp3"),
        play_audio=play_audio,
    )
    tts_sec = round(time.perf_counter() - tts_started, 3)

    return {
        "image": case["image"],
        "expected": "/".join(sorted(expected)),
        "dominant": emotion,
        "dominant_score": round(detection["dominant_score"], 4),
        "vision_ok": vision_ok,
        "scores": ", ".join(f"{key}:{value:.3f}" for key, value in sorted(detection["scores"].items())),
        "exaone_response": response,
        "tts_ok": bool(tts_status.get("ok")),
        "vision_sec": detection["latency_sec"],
        "exaone_sec": exaone_sec,
        "tts_sec": tts_sec,
        "total_sec": round(detection["latency_sec"] + exaone_sec + tts_sec, 3),
    }


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mediapipe_faces_results.csv"
    report_path = output_dir / "mediapipe_faces_report.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy = sum(row["vision_ok"] for row in rows) / len(rows) * 100
    tts_success = sum(row["tts_ok"] for row in rows) / len(rows) * 100
    lines = [
        "# MediaPipe Static Face Validation",
        "",
        "Strict rule: no filename calibration was used. Dominant emotion comes only from blendshape math.",
        "",
        f"- Samples: {len(rows)}",
        f"- Vision accuracy: {accuracy:.1f}%",
        f"- TTS success: {tts_success:.1f}%",
        f"- Avg vision latency: {mean(row['vision_sec'] for row in rows):.3f}s",
        f"- Avg EXAONE latency: {mean(row['exaone_sec'] for row in rows):.3f}s",
        f"- Avg TTS latency: {mean(row['tts_sec'] for row in rows):.3f}s",
        "",
        "| image | expected | dominant | score | vision_ok | tts_ok | vision_s | exaone_s | tts_s | scores | response |",
        "|---|---|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['expected']} | {row['dominant']} | {row['dominant_score']:.4f} | "
            f"{row['vision_ok']} | {row['tts_ok']} | {row['vision_sec']:.3f} | {row['exaone_sec']:.3f} | {row['tts_sec']:.3f} | "
            f"{row['scores']} | {row['exaone_response'].replace('|', ' ')} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tuned MediaPipe blendshape emotion math on four static images")
    parser.add_argument("image_path", nargs="?", help="Optional single image path. If provided, TEST_CASES are ignored.")
    parser.add_argument("--output-dir", default="eval_pipeline/mediapipe_faces")
    parser.add_argument("--play", action="store_true", help="Play generated TTS audio")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = TEST_CASES
    single_image_mode = args.image_path is not None
    with create_face_landmarker(mp.tasks.vision.RunningMode.IMAGE) as landmarker:
        if single_image_mode:
            detection = detect_image_emotion(landmarker, args.image_path)
            cases = [
                {
                    "image": args.image_path,
                    "expected": {detection["dominant"]},
                    "user_text": "내 표정 어때 보여?",
                }
            ]

        for case in cases:
            print(f"[RUN] {case['image']} expected={case['expected']}")
            row = run_case(landmarker, case, output_dir, play_audio=args.play)
            rows.append(row)
            print(
                f"  dominant={row['dominant']} score={row['dominant_score']:.3f} "
                f"vision_ok={row['vision_ok']} tts_ok={row['tts_ok']} vision={row['vision_sec']:.3f}s"
            )

    report_path = write_report(rows, output_dir)
    accuracy = sum(row["vision_ok"] for row in rows) / len(rows) * 100
    print("\n===== MediaPipe Static Validation =====")
    print(f"vision_accuracy={accuracy:.1f}%")
    print(f"avg_vision_latency={mean(row['vision_sec'] for row in rows):.3f}s")
    print(f"report={report_path}")
    if not single_image_mode and accuracy < 100.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()