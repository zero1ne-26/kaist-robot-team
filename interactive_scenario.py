from __future__ import annotations

import argparse
import enum
import time
from typing import Any

import requests

from loop4 import (
    KEEP_ALIVE,
    OLLAMA_CHAT_URL,
    REASONING_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    sanitize_tts_text,
    warm_up_models,
)
from audio_manager import AudioManager


INITIAL_SAD_QUESTION = "오늘 표정이 슬퍼보이는데 무슨 일 있으세요?"

EXAONE_HRI_SYSTEM_PROMPT = """너는 자비스라는 한국어 로봇 비서다.
너의 성격은 가까운 친구처럼 따뜻하고, 로봇 비서처럼 차분하며, 먼저 힘을 주는 정서 지능형 HRI 파트너다.

반드시 지킬 응답 구조:
1. 사용자의 말을 구체적으로 반영한다. 시험, 프로젝트, 싸움, 놀란 일처럼 핵심 사건을 직접 언급한다.
2. 감정을 인정한다. "정말 속상했겠어요", "그건 뿌듯하겠어요"처럼 마음을 먼저 받아준다.
3. 앞으로의 힘을 준다. 다음 행동, 회복, 축하 질문, 격려 중 하나를 반드시 덧붙인다.
4. 한국어 1~2문장만 말한다. 이모지, markdown, 분석 라벨, 영어 설명은 금지한다.
5. 사용자가 힘들어할 때는 절대 훈계하지 말고, "다음번엔 꼭 더 잘하실 수 있을 거예요", "제가 옆에서 도울게요"처럼 적극적으로 격려한다.

감정별 행동 규칙:
- sad: "무슨 일 있으세요?"를 반복하지 말고, 이미 말한 이유를 반영해 깊게 위로하고 회복을 격려한다.
- happy: 걱정 질문 금지. 함께 기뻐하고, 더 들려달라고 자연스럽게 묻는다.
- angry: 화난 이유를 인정하고, 편을 들어주되 차분해질 작은 다음 행동을 제안한다.
- surprised: 놀란 상황을 받아주고, 괜찮은지 확인하며 자세히 말해달라고 한다.
- neutral: 부담 없이 인사하고, 필요한 일이 있으면 돕겠다고 말한다.

좋은 예시:
사용자: 오늘 시험을 망쳤어!!
자비스: 시험을 망쳐서 정말 속상했겠어요. 그래도 오늘 하나로 끝난 건 아니니까, 다음번엔 꼭 더 잘하실 수 있을 거예요. 힘내세요!
사용자: 오늘 드디어 프로젝트를 끝냈어!
자비스: 드디어 프로젝트를 끝냈다니 정말 뿌듯하겠어요. 고생 많았어요, 어떤 부분이 제일 만족스러웠는지도 저랑 공유해요!
"""


class HRIState(enum.Enum):
    OBSERVING = "IDLING/OBSERVING"
    PROACTIVE_INITIATION = "PROACTIVE_INITIATION"
    LISTENING = "LISTENING(STT)"
    RESPONDING = "RESPONDING"
    DONE = "DONE"


def capture_frame() -> dict[str, Any]:
    """Mock camera/VLM hook. Replace this with real Moondream frame analysis later."""
    time.sleep(0.1)
    return {
        "emotion": "sad",
        "description": "The user looks sad and gloomy, with a downcast facial expression.",
        "confidence": 0.92,
    }


def listen_stt() -> str:
    """Mock STT hook. Terminal input simulates the user's recognized speech."""
    user_text = input("[STT 입력 시뮬레이션] 사용자 말: ").strip()
    return user_text or "오늘 시험을 망쳤어!!"


def speak_tts(text: str, output_path: str = "interactive_response.mp3", play_audio: bool = True) -> dict[str, Any]:
    """Speak sanitized Korean text using the existing local TTS module."""
    safe_text = sanitize_tts_text(text)
    manager = AudioManager(model=REASONING_MODEL)
    started = time.perf_counter()
    audio_path = manager.synthesize_speech(safe_text, output_path)
    if play_audio:
        manager._play_audio_file(audio_path)
    return {
        "ok": True,
        "text": safe_text,
        "audio_path": audio_path,
        "played": play_audio,
        "latency_sec": round(time.perf_counter() - started, 3),
    }


def build_contextual_messages(
    conversation_history: list[dict[str, str]],
    vision_description: str,
    emotion: str = "sad",
) -> list[dict[str, str]]:
    """Combine system prompt, visual context, and multi-turn history for EXAONE."""
    system_message = next(
        (turn for turn in conversation_history if turn.get("role") == "system"),
        {"role": "system", "content": EXAONE_HRI_SYSTEM_PROMPT},
    )
    dialogue_turns = [turn for turn in conversation_history if turn.get("role") != "system"]
    messages = [
        system_message,
        {
            "role": "user",
            "content": (
                "현재 시각 맥락: "
                f"{vision_description}\n"
                f"감정 라벨: {emotion}\n"
                "이전 대화와 사용자의 방금 발화를 바탕으로 로봇이 지금 말할 문장만 생성해. "
                "응답에는 상황 반영, 감정 인정, 앞으로의 격려를 모두 포함해."
            ),
        },
    ]
    messages.extend(dialogue_turns)
    return messages


def call_exaone_contextual_response(
    conversation_history: list[dict[str, str]],
    vision_description: str,
    emotion: str = "sad",
) -> str:
    """Generate the second-turn empathetic response with EXAONE and keep_alive enabled."""
    payload = {
        "model": REASONING_MODEL,
        "messages": build_contextual_messages(conversation_history, vision_description, emotion=emotion),
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.25,
            "num_predict": 96,
            "num_ctx": 1024,
        },
    }
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError("EXAONE 응답 시간이 초과되었습니다.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"EXAONE 호출 실패: {exc}") from exc

    text = str(response.json().get("message", {}).get("content", "")).strip()
    if not text:
        raise RuntimeError("EXAONE이 빈 응답을 반환했습니다.")
    return sanitize_tts_text(text)


def run_interactive_sad_exam_scenario(play_audio: bool = True, warmup: bool = True) -> dict[str, Any]:
    """Run the exact proactive sad-face -> STT -> contextual empathy scenario."""
    if warmup:
        print("[WARMUP] Ollama 모델을 메모리에 유지합니다.")
        warm_up_models()

    state = HRIState.OBSERVING
    conversation_history: list[dict[str, str]] = [
        {"role": "system", "content": EXAONE_HRI_SYSTEM_PROMPT}
    ]
    result: dict[str, Any] = {
        "vision": None,
        "initial_tts": None,
        "user_text": "",
        "final_response": "",
        "final_tts": None,
        "conversation_history": conversation_history,
    }

    while state is not HRIState.DONE:
        print(f"[STATE] {state.value}")

        if state is HRIState.OBSERVING:
            vision = capture_frame()
            result["vision"] = vision
            print(f"[VISION] emotion={vision['emotion']} description={vision['description']}")
            state = HRIState.PROACTIVE_INITIATION if vision.get("emotion") == "sad" else HRIState.DONE

        elif state is HRIState.PROACTIVE_INITIATION:
            print(f"[JARVIS-1] {INITIAL_SAD_QUESTION}")
            result["initial_tts"] = speak_tts(
                INITIAL_SAD_QUESTION,
                output_path="eval_pipeline/interactive_initial.mp3",
                play_audio=play_audio,
            )
            conversation_history.append({"role": "assistant", "content": INITIAL_SAD_QUESTION})
            state = HRIState.LISTENING

        elif state is HRIState.LISTENING:
            user_text = listen_stt()
            print(f"[USER/STT] {user_text}")
            result["user_text"] = user_text
            conversation_history.append({"role": "user", "content": user_text})
            state = HRIState.RESPONDING

        elif state is HRIState.RESPONDING:
            vision = result["vision"] or {"description": "사용자가 슬퍼 보임.", "emotion": "sad"}
            final_response = call_exaone_contextual_response(
                conversation_history,
                vision["description"],
                emotion=vision.get("emotion", "sad"),
            )
            print(f"[JARVIS-2] {final_response}")
            result["final_response"] = final_response
            result["final_tts"] = speak_tts(
                final_response,
                output_path="eval_pipeline/interactive_final.mp3",
                play_audio=play_audio,
            )
            conversation_history.append({"role": "assistant", "content": final_response})
            state = HRIState.DONE

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Proactive multi-turn HRI scenario: sad face -> STT -> empathetic response")
    parser.add_argument("--no-play", action="store_true", help="Synthesize TTS files without playing audio")
    parser.add_argument("--no-warmup", action="store_true", help="Skip Ollama keep-alive warm-up")
    args = parser.parse_args()

    result = run_interactive_sad_exam_scenario(play_audio=not args.no_play, warmup=not args.no_warmup)
    print("\n[CONVERSATION HISTORY]")
    for turn in result["conversation_history"]:
        print(f"- {turn['role']}: {turn['content']}")


if __name__ == "__main__":
    main()