from __future__ import annotations

import argparse
import enum
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from utils.loop4 import warm_up_models
from utils.hri_response import EXAONE_HRI_SYSTEM_PROMPT, call_exaone_contextual_response
from utils.mediapipe_bridge import EmotionEngine, create_face_landmarker, draw_ai_view, emotion_scores_from_result
from utils.stt import listen_from_terminal
from utils.tts import speak_tts


TRIGGER_EMOTIONS = {"sad", "angry", "fear"}
DEFAULT_TRIGGER_THRESHOLD = 0.70
DEFAULT_CONSECUTIVE_FRAMES = 3

PROACTIVE_PROMPTS = {
    "sad": "표정이 안 좋아 보이시는데, 무슨 일 있으세요?",
    "angry": "조금 화가 나 보이세요. 제가 차분히 도와드릴까요?",
    "fear": "불안하거나 긴장해 보이세요. 괜찮으신가요? 천천히 말씀해 주세요.",
}


class RobotState(enum.Enum):
    OBSERVING = "OBSERVING"
    PROACTIVE_INITIATION = "PROACTIVE_INITIATION"
    LISTENING = "LISTENING"
    RESPONDING = "RESPONDING"
    SHUTDOWN = "SHUTDOWN"


def listen_stt() -> str:
    return listen_from_terminal(default_text="오늘 좀 힘들어요")


def _draw_landmarks(frame, face_landmarks, width: int, height: int) -> None:
    if face_landmarks is None:
        return
    for lm in face_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def observe_until_trigger(
    camera_index: int,
    threshold: float,
    consecutive_frames: int,
) -> dict[str, Any] | None:
    """Run the real-time MediaPipe observation loop until a trigger emotion persists."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: index={camera_index}")

    start_time = time.time()
    streak_emotion: str | None = None
    streak_count = 0

    try:
        with create_face_landmarker(mp.tasks.vision.RunningMode.VIDEO) as landmarker:
            emotion_engine = EmotionEngine(window_size=10, ema_alpha=0.35)
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[OBSERVING] 카메라 프레임을 읽을 수 없습니다.")
                    return None

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                emotion_scores, face_landmarks = emotion_scores_from_result(result, engine=emotion_engine)

                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                dominant_score = float(emotion_scores[dominant_emotion])
                _draw_landmarks(frame, face_landmarks, width, height)

                if dominant_emotion in TRIGGER_EMOTIONS and dominant_score >= threshold:
                    if streak_emotion == dominant_emotion:
                        streak_count += 1
                    else:
                        streak_emotion = dominant_emotion
                        streak_count = 1
                else:
                    streak_emotion = None
                    streak_count = 0

                display_text = f"{dominant_emotion.upper()} {dominant_score * 100:.1f}% | streak {streak_count}/{consecutive_frames}"
                cv2.putText(frame, display_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                ai_view = draw_ai_view(face_landmarks, emotion_scores, width, height)
                cv2.imshow("Human View - Emotion Output", frame)
                cv2.imshow("AI Monitoring View - Facial Mesh & Emotion Bars", ai_view)

                if streak_emotion and streak_count >= consecutive_frames:
                    return {
                        "emotion": streak_emotion,
                        "score": dominant_score,
                        "scores": emotion_scores,
                        "description": f"MediaPipe detected {streak_emotion} for {streak_count} consecutive frames at score {dominant_score:.2f}.",
                    }

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def generate_proactive_message(emotion: str, conversation_history: list[dict[str, str]]) -> str:
    """Use EXAONE for the first proactive message, with a deterministic fallback."""
    fallback = PROACTIVE_PROMPTS.get(emotion, "표정이 평소와 조금 달라 보여요. 괜찮으세요?")
    history = conversation_history + [
        {"role": "user", "content": f"사용자의 얼굴에서 {emotion} 감정이 강하게 감지되었습니다. 먼저 말을 걸어주세요."}
    ]
    try:
        response = call_exaone_contextual_response(
            history,
            vision_description=f"MediaPipe dominant emotion: {emotion}",
            emotion=emotion,
        )
        return response or fallback
    except Exception as exc:
        print(f"[WARN] EXAONE proactive response failed: {exc}")
        return fallback


def run_robot_loop(
    camera_index: int = 0,
    threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    consecutive_frames: int = DEFAULT_CONSECUTIVE_FRAMES,
    play_audio: bool = True,
    warmup: bool = True,
    max_cycles: int = 0,
) -> None:
    if warmup:
        print("[WARMUP] Ollama 모델을 메모리에 유지합니다.")
        warm_up_models()

    state = RobotState.OBSERVING
    cycle_count = 0
    latest_vision: dict[str, Any] | None = None
    latest_user_text = ""
    conversation_history: list[dict[str, str]] = [
        {"role": "system", "content": EXAONE_HRI_SYSTEM_PROMPT}
    ]

    while state is not RobotState.SHUTDOWN:
        print(f"[STATE] {state.value}")

        if state is RobotState.OBSERVING:
            if max_cycles and cycle_count >= max_cycles:
                state = RobotState.SHUTDOWN
                continue
            latest_vision = observe_until_trigger(camera_index, threshold, consecutive_frames)
            if latest_vision is None:
                state = RobotState.SHUTDOWN
            else:
                print(f"[TRIGGER] {latest_vision['emotion']} score={latest_vision['score']:.2f}")
                state = RobotState.PROACTIVE_INITIATION

        elif state is RobotState.PROACTIVE_INITIATION:
            emotion = latest_vision["emotion"] if latest_vision else "sad"
            proactive_text = generate_proactive_message(emotion, conversation_history)
            print(f"[JARVIS-1] {proactive_text}")
            speak_tts(
                proactive_text,
                output_path=f"eval_pipeline/multimodal_{cycle_count + 1}_initial.mp3",
                play_audio=play_audio,
            )
            conversation_history.append({"role": "assistant", "content": proactive_text})
            state = RobotState.LISTENING

        elif state is RobotState.LISTENING:
            latest_user_text = listen_stt()
            print(f"[USER/STT] {latest_user_text}")
            conversation_history.append({"role": "user", "content": latest_user_text})
            state = RobotState.RESPONDING

        elif state is RobotState.RESPONDING:
            emotion = latest_vision["emotion"] if latest_vision else "sad"
            vision_description = latest_vision["description"] if latest_vision else "사용자의 부정적 감정이 감지됨."
            final_response = call_exaone_contextual_response(conversation_history, vision_description, emotion=emotion)
            print(f"[JARVIS-2] {final_response}")
            speak_tts(
                final_response,
                output_path=f"eval_pipeline/multimodal_{cycle_count + 1}_final.mp3",
                play_audio=play_audio,
            )
            conversation_history.append({"role": "assistant", "content": final_response})
            cycle_count += 1
            state = RobotState.OBSERVING


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time MediaPipe blendshape multimodal robot loop")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD)
    parser.add_argument("--consecutive-frames", type=int, default=DEFAULT_CONSECUTIVE_FRAMES)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run until q/camera stop")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    run_robot_loop(
        camera_index=args.camera_index,
        threshold=args.threshold,
        consecutive_frames=args.consecutive_frames,
        play_audio=not args.no_play,
        warmup=not args.no_warmup,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()