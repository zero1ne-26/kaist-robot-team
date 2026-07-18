from __future__ import annotations

import argparse
from pathlib import Path

from loop4 import run_interaction_pipeline


DEFAULT_TEST_IMAGES = {
    "happy": "happy_face.jpg",
    "sad": "sad_face.jpg",
    "neutral": "neutral_face.jpg",
}

EMPATHY_CHECK_INS = (
    "무슨 일 있으세요",
    "괜찮으세요",
    "힘든 일",
    "속상",
    "도와드릴까요",
)


def _looks_gloomy(description: str) -> bool:
    lowered = (description or "").lower()
    return any(keyword in lowered for keyword in ("sad", "gloomy", "upset", "angry", "unhappy", "down"))


def _assert_sad_case_has_empathy(case_name: str, moondream_raw: str, exaone_text: str) -> None:
    should_validate = case_name == "sad" or _looks_gloomy(moondream_raw)
    if not should_validate:
        return
    if not any(phrase in exaone_text for phrase in EMPATHY_CHECK_INS):
        raise AssertionError(
            "sad/gloomy 케이스에서 공감 확인 문구가 나오지 않았습니다. "
            f"EXAONE output={exaone_text!r}"
        )


def run_test_case(case_name: str, image_path: Path, output_dir: Path, play_audio: bool) -> dict:
    output_path = output_dir / f"{case_name}_response.mp3"
    result = run_interaction_pipeline(image_path, output_path=str(output_path), play_audio=play_audio)

    print(f"\n===== TEST CASE: {case_name} | {image_path} =====")
    print(f"[STAGE 1] Moondream Raw Output: {result['moondream_raw']}")
    print(f"[STAGE 2] EXAONE Decision & Text: {result['exaone_text']}")
    print(f"[STAGE 3] TTS Execution Status: {result['tts_status']}")
    print(f"[TIMINGS] {result['timings']}")

    if not result["success"]:
        raise RuntimeError(f"pipeline failed for {case_name}: {result['error']}")
    if not result["tts_status"].get("ok"):
        raise AssertionError(f"TTS execution failed for {case_name}")

    _assert_sad_case_has_empathy(case_name, result["moondream_raw"], result["exaone_text"])
    print(f"[ASSERTION] {case_name}: PASS")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="QA harness for Moondream -> EXAONE -> TTS facial-expression pipeline")
    parser.add_argument("--image-dir", default="test_images", help="Directory containing happy_face.jpg, sad_face.jpg, neutral_face.jpg")
    parser.add_argument("--output-dir", default="eval_pipeline", help="Directory for generated TTS MP3 files")
    parser.add_argument("--play", action="store_true", help="Play each generated TTS file during the test")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_cases = {
        case_name: image_dir / filename
        for case_name, filename in DEFAULT_TEST_IMAGES.items()
        if (image_dir / filename).exists()
    }
    if not discovered_cases:
        expected = ", ".join(DEFAULT_TEST_IMAGES.values())
        raise SystemExit(f"테스트 이미지가 없습니다. {image_dir}/ 아래에 다음 파일을 넣어주세요: {expected}")

    failures: list[str] = []
    for case_name, image_path in discovered_cases.items():
        try:
            run_test_case(case_name, image_path, output_dir, play_audio=args.play)
        except Exception as exc:
            failures.append(f"{case_name}: {exc}")
            print(f"[ASSERTION] {case_name}: FAIL - {exc}")

    print("\n===== QA SUMMARY =====")
    print(f"total={len(discovered_cases)} passed={len(discovered_cases) - len(failures)} failed={len(failures)}")
    if failures:
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()