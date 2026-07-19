import argparse
import signal
from pathlib import Path

from utils.audio_manager import AudioManager
from utils.states import ConversationStateMachine

DEFAULT_MIC_TIMEOUT = 10.0
DEFAULT_MIC_PHRASE_TIME_LIMIT = 10.0
DEFAULT_SYSTEM_PROMPT = (
    "너는 한국어 음성비서야. "
    "사용자의 질문에 직접 답해. "
    "답변은 한국어 1~2문장만 말해. "
    "지시를 이해했다는 말, 예시 질문, 후속 질문 추천은 말하지 마. "
    "날씨, 알람, 검색, 기기 제어 같은 도구 요청은 코드가 먼저 처리하므로 일반 대화에서는 함수 호출 JSON을 만들지 마. "
    "미래 예언, 존재하지 않는 사실, 개인 비밀처럼 알 수 없는 질문만 절대 지어내지 말고, 반드시 '헤헤, 그건 지니가 아직 공부를 못해서 잘 몰라!'라고 자연스럽게 대답해라."
)


def create_pipeline(
    output_path: str = "response.mp3",
    base_url: str | None = None,
    api_key: str = "dummy",
    model: str = "exaone3.5:2.4b",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    input_type: str = "mic",
    file_path: str | None = None,
    mic_timeout: float = DEFAULT_MIC_TIMEOUT,
    mic_phrase_time_limit: float = DEFAULT_MIC_PHRASE_TIME_LIMIT,
) -> ConversationStateMachine:
    output_file = Path(output_path)
    output_dir = output_file.parent if output_file.parent != Path("") else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_manager = AudioManager(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
    )
    machine = ConversationStateMachine(
        audio_manager=audio_manager,
        output_path=output_file,
        input_type=input_type,
        file_path=file_path,
        mic_timeout=mic_timeout,
        mic_phrase_time_limit=mic_phrase_time_limit,
    )

    return machine


def run_pipeline(
    output_path: str = "response.mp3",
    base_url: str | None = None,
    api_key: str = "dummy",
    model: str = "exaone3.5:2.4b",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    input_type: str = "mic",
    file_path: str | None = None,
    mic_timeout: float = DEFAULT_MIC_TIMEOUT,
    mic_phrase_time_limit: float = DEFAULT_MIC_PHRASE_TIME_LIMIT,
) -> dict:
    machine = create_pipeline(
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        input_type=input_type,
        file_path=file_path,
        mic_timeout=mic_timeout,
        mic_phrase_time_limit=mic_phrase_time_limit,
    )
    return machine.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="상태 기계 기반 STT-LLM-TTS 파이프라인")
    parser.add_argument("-o", "--output", default="response.mp3", help="저장할 TTS 오디오 파일 경로")
    parser.add_argument(
        "--input-type",
        choices=["mic", "file"],
        default="mic",
        help="STT 입력 타입 선택 (mic 또는 file)",
    )
    parser.add_argument(
        "--file-path",
        default=None,
        help="input-type이 file일 때 사용할 오디오 파일 경로",
    )
    parser.add_argument(
        "--mic-timeout",
        type=float,
        default=DEFAULT_MIC_TIMEOUT,
        help="마이크 입력 시작 대기 시간(초)",
    )
    parser.add_argument(
        "--mic-phrase-time-limit",
        type=float,
        default=DEFAULT_MIC_PHRASE_TIME_LIMIT,
        help="한 번에 녹음할 최대 음성 길이(초)",
    )
    args = parser.parse_args()

    if args.input_type == "file" and not args.file_path:
        parser.error("--input-type file 사용 시 --file-path가 필요합니다.")

    machine = create_pipeline(
        output_path=args.output,
        input_type=args.input_type,
        file_path=args.file_path,
        mic_timeout=args.mic_timeout,
        mic_phrase_time_limit=args.mic_phrase_time_limit,
    )

    def _handle_shutdown(signum, _frame) -> None:
        print(f"[SYSTEM] 수신된 종료 신호({signum})를 처리합니다.", flush=True)
        machine.shutdown()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        machine.run()
    except KeyboardInterrupt:
        machine.shutdown()


if __name__ == "__main__":
    main()
