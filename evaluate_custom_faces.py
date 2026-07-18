from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean
from typing import Any

from loop4 import (
    get_exaone_empathy_response,
    play_tts,
    describe_user_emotion_details,
    warm_up_models,
)


CUSTOM_FACE_CASES = [
    {"image": "sad_face.jpg", "expected": "Sad", "user_text": "오늘 표정이 좀 안 좋아 보여"},
    {"image": "mad.jpg", "expected": "Angry", "user_text": "지금 너무 화가 나"},
    {"image": "happt.jpg", "expected": "Happy", "user_text": "오늘 기분이 정말 좋아"},
    {"image": "anxi.jpg", "expected": "Anxious", "user_text": "괜히 불안하고 긴장돼"},
]

EXPECTED_ALIASES = {
    "Sad": {"Sad"},
    "Angry": {"Angry"},
    "Happy": {"Happy"},
    "Anxious": {"Anxious", "Nervous"},
}

RESPONSE_KEYWORDS = {
    "Sad": ("무슨 일", "속상", "괜찮", "힘내", "도울"),
    "Angry": ("화", "차분", "괜찮", "도울", "천천히"),
    "Happy": ("좋", "기쁘", "축하", "공유", "멋지"),
    "Anxious": ("불안", "긴장", "괜찮", "천천히", "숨", "안심"),
}


def response_matches_emotion(expected: str, response: str) -> bool:
    return any(keyword in response for keyword in RESPONSE_KEYWORDS.get(expected, ()))


def run_case(case: dict[str, str], output_dir: Path, play_audio: bool) -> dict[str, Any]:
    image_path = Path(case["image"])
    expected = case["expected"]
    if not image_path.exists():
        return {
            "image": str(image_path),
            "expected": expected,
            "moondream_raw": "",
            "zero_shot_emotion": "missing_file",
            "moondream_emotion": "missing_file",
            "calibration_applied": False,
            "roi_path": "",
            "zero_shot_ok": False,
            "vision_ok": False,
            "exaone_text": "",
            "response_ok": False,
            "tts_ok": False,
            "moondream_sec": 0.0,
            "exaone_sec": 0.0,
            "tts_sec": 0.0,
            "total_sec": 0.0,
            "error": f"missing file: {image_path}",
        }

    total_started = time.perf_counter()
    error = ""

    stage_started = time.perf_counter()
    try:
        vision = describe_user_emotion_details(image_path)
    except Exception as exc:
        vision = {"raw": "", "emotion": "error", "roi_path": ""}
        error = f"vision_error={type(exc).__name__}: {exc}"
    moondream_sec = round(time.perf_counter() - stage_started, 3)

    stage_started = time.perf_counter()
    try:
        exaone_text = get_exaone_empathy_response(vision["emotion"], user_text=case["user_text"])
    except Exception as exc:
        exaone_text = ""
        error = f"{error}; exaone_error={type(exc).__name__}: {exc}".strip("; ")
    exaone_sec = round(time.perf_counter() - stage_started, 3)

    stage_started = time.perf_counter()
    try:
        tts_status = play_tts(
            exaone_text,
            output_path=str(output_dir / f"{image_path.stem}_response.mp3"),
            play_audio=play_audio,
        ) if exaone_text else {"ok": False, "latency_sec": 0.0, "audio_path": None}
    except Exception as exc:
        tts_status = {"ok": False, "latency_sec": 0.0, "audio_path": None}
        error = f"{error}; tts_error={type(exc).__name__}: {exc}".strip("; ")
    tts_sec = round(time.perf_counter() - stage_started, 3)

    emotion = str(vision["emotion"])
    zero_shot_emotion = str(vision.get("zero_shot_emotion", emotion))
    zero_shot_ok = zero_shot_emotion in EXPECTED_ALIASES.get(expected, {expected})
    vision_ok = emotion in EXPECTED_ALIASES.get(expected, {expected})
    response_ok = response_matches_emotion(expected, exaone_text)
    return {
        "image": str(image_path),
        "expected": expected,
        "moondream_raw": vision["raw"],
        "zero_shot_emotion": zero_shot_emotion,
        "moondream_emotion": emotion,
        "calibration_applied": bool(vision.get("calibration_applied", False)),
        "roi_path": vision["roi_path"],
        "zero_shot_ok": zero_shot_ok,
        "vision_ok": vision_ok,
        "exaone_text": exaone_text,
        "response_ok": response_ok,
        "tts_ok": bool(tts_status.get("ok")),
        "tts_audio_path": tts_status.get("audio_path"),
        "moondream_sec": moondream_sec,
        "exaone_sec": exaone_sec,
        "tts_sec": tts_sec,
        "total_sec": round(time.perf_counter() - total_started, 3),
        "error": error,
    }


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "custom_faces_results.csv"
    md_path = output_dir / "custom_faces_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    zero_shot_acc = sum(row["zero_shot_ok"] for row in rows) / len(rows) * 100
    vision_acc = sum(row["vision_ok"] for row in rows) / len(rows) * 100
    response_acc = sum(row["response_ok"] for row in rows) / len(rows) * 100
    tts_acc = sum(row["tts_ok"] for row in rows) / len(rows) * 100
    lines = [
        "# Custom Face Evaluation",
        "",
        f"- Samples: {len(rows)}",
        f"- Moondream zero-shot accuracy: {zero_shot_acc:.1f}%",
        f"- Vision accuracy: {vision_acc:.1f}%",
        f"- EXAONE response match: {response_acc:.1f}%",
        f"- TTS success: {tts_acc:.1f}%",
        f"- Avg Moondream latency: {mean(row['moondream_sec'] for row in rows):.3f}s",
        f"- Avg EXAONE latency: {mean(row['exaone_sec'] for row in rows):.3f}s",
        f"- Avg TTS latency: {mean(row['tts_sec'] for row in rows):.3f}s",
        f"- Avg total latency: {mean(row['total_sec'] for row in rows):.3f}s",
        "",
        "| image | expected | raw | zero_shot | final | calibrated | zero_ok | final_ok | exaone | response_ok | tts_ok | moondream_s | exaone_s | tts_s | total_s | roi | error |",
        "|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['expected']} | {row['moondream_raw']} | {row['zero_shot_emotion']} | {row['moondream_emotion']} | "
            f"{row['calibration_applied']} | {row['zero_shot_ok']} | {row['vision_ok']} | {row['exaone_text'].replace('|', ' ')} | {row['response_ok']} | {row['tts_ok']} | "
            f"{row['moondream_sec']:.3f} | {row['exaone_sec']:.3f} | {row['tts_sec']:.3f} | {row['total_sec']:.3f} | "
            f"{row['roi_path']} | {row['error']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate four custom face images through Moondream -> EXAONE -> TTS")
    parser.add_argument("--output-dir", default="eval_pipeline/custom_faces", help="Report and TTS output directory")
    parser.add_argument("--play", action="store_true", help="Play generated TTS audio")
    parser.add_argument("--no-warmup", action="store_true", help="Skip Ollama keep-alive warm-up")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_warmup:
        warm_up_models()

    rows = []
    for case in CUSTOM_FACE_CASES:
        print(f"[RUN] {case['image']} expected={case['expected']}")
        row = run_case(case, output_dir, play_audio=args.play)
        rows.append(row)
        print(
            f"  raw={row['moondream_raw']!r} zero_shot={row['zero_shot_emotion']} final={row['moondream_emotion']} "
            f"calibrated={row['calibration_applied']} "
            f"vision_ok={row['vision_ok']} response_ok={row['response_ok']} tts_ok={row['tts_ok']} "
            f"latency={row['total_sec']:.3f}s"
        )

    report_path = write_report(rows, output_dir)
    print("\n===== Custom Face Scorecard =====")
    print(f"zero_shot_accuracy={sum(row['zero_shot_ok'] for row in rows) / len(rows) * 100:.1f}%")
    print(f"vision_accuracy={sum(row['vision_ok'] for row in rows) / len(rows) * 100:.1f}%")
    print(f"response_match={sum(row['response_ok'] for row in rows) / len(rows) * 100:.1f}%")
    print(f"tts_success={sum(row['tts_ok'] for row in rows) / len(rows) * 100:.1f}%")
    print(f"avg_total_latency={mean(row['total_sec'] for row in rows):.3f}s")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()