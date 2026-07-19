import argparse

from utils.audio_manager import AudioManager

# Adjust these values as needed.
LISTEN_TIMEOUT_SECONDS = 10.0
PHRASE_TIME_LIMIT_SECONDS = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="마이크 단독 STT 테스트")
    parser.add_argument(
        "--timeout",
        type=float,
        default=LISTEN_TIMEOUT_SECONDS,
        help="마이크 입력 시작 대기 시간(초)",
    )
    parser.add_argument(
        "--phrase-time-limit",
        type=float,
        default=PHRASE_TIME_LIMIT_SECONDS,
        help="한 번에 녹음할 최대 음성 길이(초)",
    )
    args = parser.parse_args()

    manager = AudioManager()
    print(f"[Test] 최대 {args.phrase_time_limit}초 동안 마이크 입력을 테스트합니다.", flush=True)

    text = manager.transcribe_from_microphone(
        timeout=args.timeout,
        phrase_time_limit=args.phrase_time_limit,
    )

    if text.strip():
        print(f"[Test STT 결과] : {text}", flush=True)
    else:
        print("[Test STT 결과] : 인식된 텍스트가 없습니다.", flush=True)


if __name__ == "__main__":
    main()
