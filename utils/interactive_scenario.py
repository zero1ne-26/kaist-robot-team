from __future__ import annotations

import argparse
import enum
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.loop4 import warm_up_models
from utils.hri_response import EXAONE_HRI_SYSTEM_PROMPT, call_exaone_contextual_response
from utils.stt import listen_from_terminal
from utils.tts import speak_tts


INITIAL_SAD_QUESTION = "오늘 표정이 슬퍼보이는데 무슨 일 있으세요?"

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
    return listen_from_terminal(default_text="오늘 시험을 망쳤어!!")


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