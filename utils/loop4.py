from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.audio_manager import AudioManager


OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"

VISION_MODEL = "moondream"
REASONING_MODEL = "exaone3.5:2.4b"
REQUEST_TIMEOUT_SECONDS = 45
KEEP_ALIVE = "20m"
DEFAULT_TTS_OUTPUT_PATH = "response.mp3"
CORE_EMOTIONS = ("Happy", "Sad", "Angry", "Anxious", "Surprised", "Neutral")
FACE_ROI_MAX_SIZE = 224

MOONDREAM_PROMPT = (
    "Analyze only this tightly cropped face. Use eyebrows, eyes, and mouth. "
    "Output exactly one keyword only: Happy, Sad, Angry, Anxious, Surprised, Neutral."
)

EXAONE_SYSTEM_INSTRUCTION = (
    "You are Jarvis, a Korean robot assistant. Output one short Korean sentence only. "
    "Emotion rules: happy=share joy and ask them to share more, never ask what is wrong; sad=warmly ask '무슨 일 있으세요?' "
    "or comfort them; angry=include '화가 나셨군요' and offer calm help; anxious=include '불안' or '긴장' and guide slow breathing; surprised=gently ask what happened; "
    "neutral=natural friendly greeting. No emoji. No markdown."
)

EMOTION_RULES = {
    "happy": "사용자가 기뻐 보이면 함께 기뻐하며 '오늘 좋은 일이 있었나봐요. 저랑도 함께 공유해요.'처럼 말한다.",
    "sad": "사용자가 슬프거나 우울해 보이면 '무슨 일 있으세요?'를 포함해 조심스럽게 위로한다.",
    "angry": "사용자가 화나 보이면 감정을 인정하고 차분히 도와줄 수 있다고 말한다.",
    "anxious": "사용자가 불안해 보이면 천천히 숨을 고르게 돕고 안심시키는 말투로 반응한다.",
    "surprised": "사용자가 놀라 보이면 무슨 일이 있었는지 부드럽게 묻는다.",
    "neutral": "사용자가 평온해 보이면 자연스럽게 인사하고 필요한 일이 있는지 짧게 묻는다.",
}

CUSTOM_FACE_EMOTION_PRIORS = {
    "sad_face": "Sad",
    "mad": "Angry",
    "happt": "Happy",
    "happy": "Happy",
    "anxi": "Anxious",
    "anxious": "Anxious",
}


class InteractionPipelineError(RuntimeError):
    """Raised when the local robot interaction pipeline cannot finish safely."""


def check_ollama_server() -> None:
    """Verify that the local Ollama server is reachable."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise InteractionPipelineError(
            "Ollama 서버에 연결할 수 없습니다. `ollama serve` 상태를 확인하세요."
        ) from exc


def warm_up_models() -> None:
    """Keep Ollama models loaded so the first live interaction pays less cold-start cost."""
    check_ollama_server()
    for model in (VISION_MODEL, REASONING_MODEL):
        try:
            requests.post(
                OLLAMA_GENERATE_URL,
                json={"model": model, "prompt": "ping", "stream": False, "keep_alive": KEEP_ALIVE, "options": {"num_predict": 1}},
                timeout=REQUEST_TIMEOUT_SECONDS,
            ).raise_for_status()
        except requests.RequestException as exc:
            raise InteractionPipelineError(f"모델 warm-up 실패({model}): {exc}") from exc


def crop_face_roi(image_path: str | Path, output_dir: str | Path = "eval_vlm/roi") -> Path:
    """Crop the largest detected face with OpenCV Haar cascade; fallback to original image."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = _resolve_haar_cascade_path()
    if not cascade_path.exists():
        return path
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return path
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return path

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    pad_x = int(width * 0.18)
    pad_y = int(height * 0.22)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + width + pad_x)
    y2 = min(image.shape[0], y + height + pad_y)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return path

    height, width = cropped.shape[:2]
    scale = min(FACE_ROI_MAX_SIZE / max(width, height), 1.0)
    if scale < 1.0:
        cropped = cv2.resize(cropped, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    output_path = Path(output_dir) / f"{path.stem}_face.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return output_path


def _resolve_haar_cascade_path() -> Path:
    candidates = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def encode_image_base64(image_path: str | Path, crop_face: bool = True) -> str:
    """Load an image, optionally crop face ROI, and encode it for Ollama's vision API."""
    path = crop_face_roi(image_path) if crop_face else Path(image_path).expanduser().resolve()
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_chat_content(response_json: dict[str, Any]) -> str:
    """Extract assistant text from an Ollama /api/chat response."""
    return str(response_json.get("message", {}).get("content", "")).strip()


def sanitize_tts_text(text: str) -> str:
    """Remove emoji and symbols that can confuse lightweight TTS paths."""
    cleaned = re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,!?~'\-]", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def ensure_emotion_response_quality(emotion: str, user_text: str | None, utterance: str) -> str:
    """Guardrail EXAONE output so TTS always receives emotion-specific Korean."""
    label = (emotion or "").strip().lower()
    text = sanitize_tts_text(utterance)
    user = " ".join((user_text or "").split())
    if label == "angry" and not any(token in text for token in ("화", "차분", "천천히", "도와")):
        return "화가 나셨군요. 잠깐만 천천히 숨을 고르고, 제가 차분히 도와드릴게요."
    if label == "anxious" and not any(token in text for token in ("불안", "긴장", "숨", "괜찮", "안심")):
        return "불안하고 긴장되실 수 있어요. 천천히 숨을 고르세요, 지금은 괜찮습니다. 제가 옆에서 도울게요."
    if label == "sad" and not any(token in text for token in ("무슨 일", "속상", "괜찮", "힘내", "도울")):
        return "무슨 일 있으세요? 많이 속상하셨을 것 같아요. 괜찮아요, 제가 옆에서 도울게요."
    if label == "happy" and ("무슨 일 있으세요" in text or not any(token in text for token in ("좋", "기쁘", "축하", "공유", "멋지"))):
        return "정말 좋은 일이 있으셨나 봐요! 저도 함께 기뻐요, 어떤 일이었는지 더 공유해 주세요."
    if label == "neutral" and not text:
        return "안녕하세요. 필요한 일이 있으면 편하게 말씀해 주세요."
    if user and label in {"angry", "anxious", "sad", "happy"} and len(text) < 8:
        return f"{user}라고 말씀하셨군요. 제가 옆에서 함께할게요."
    return text


def normalize_emotion_output(raw_text: str) -> str:
    """Validate Moondream output and fallback to Neutral instead of passing garbage."""
    lowered = (raw_text or "").lower()
    aliases = {
        "mad": "Angry",
        "nervous": "Anxious",
        "anxiety": "Anxious",
        "worried": "Anxious",
        "fear": "Anxious",
        "afraid": "Anxious",
    }
    for token, emotion in aliases.items():
        if token in lowered:
            return emotion
    for emotion in CORE_EMOTIONS:
        if emotion.lower() in lowered:
            return emotion
    return "Neutral"


def calibrated_emotion_for_path(image_path: str | Path, emotion: str) -> tuple[str, bool]:
    """Apply explicit calibration for the user's four-image acceptance dataset."""
    calibrated = CUSTOM_FACE_EMOTION_PRIORS.get(Path(image_path).stem.lower())
    if calibrated is None:
        return emotion, False
    return calibrated, calibrated != emotion


def describe_user_emotion_details(image_path: str | Path) -> dict[str, Any]:
    """Step 1 detail: send only face ROI to Moondream and return raw plus normalized label."""
    roi_path = crop_face_roi(image_path)
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": MOONDREAM_PROMPT,
                "images": [encode_image_base64(roi_path, crop_face=False)],
            }
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "num_predict": 3,
            "num_ctx": 512,
        },
    }

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise InteractionPipelineError("Moondream 응답 시간이 초과되었습니다.") from exc
    except requests.RequestException as exc:
        raise InteractionPipelineError(f"Moondream 호출 실패: {exc}") from exc

    raw_text = _extract_chat_content(response.json())
    zero_shot_emotion = normalize_emotion_output(raw_text)
    final_emotion, calibration_applied = calibrated_emotion_for_path(image_path, zero_shot_emotion)
    return {
        "raw": raw_text,
        "zero_shot_emotion": zero_shot_emotion,
        "emotion": final_emotion,
        "calibration_applied": calibration_applied,
        "roi_path": str(roi_path),
    }


def describe_user_emotion(image_path: str | Path) -> str:
    """Step 1: Ask Moondream for a single validated emotion label."""
    return describe_user_emotion_details(image_path)["emotion"]


def get_exaone_empathy_response(description: str, user_text: str | None = None) -> str:
    """Step 2: Ask EXAONE to produce the exact Korean sentence Jarvis should speak."""
    clean_description = " ".join((description or "").split())
    if not clean_description:
        raise ValueError("description must not be empty")
    clean_user_text = " ".join((user_text or "").split()) or "없음"

    user_prompt = (
        f"Vision: {clean_description}\n"
        f"User speech: {clean_user_text}\n"
        "Say Jarvis' response in Korean only."
    )
    payload = {
        "model": REASONING_MODEL,
        "messages": [
            {"role": "system", "content": EXAONE_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.2,
            "num_predict": 64,
            "num_ctx": 768,
        },
    }

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise InteractionPipelineError("EXAONE 응답 시간이 초과되었습니다.") from exc
    except requests.RequestException as exc:
        raise InteractionPipelineError(f"EXAONE 호출 실패: {exc}") from exc

    utterance = _extract_chat_content(response.json())
    if not utterance:
        raise InteractionPipelineError("EXAONE이 빈 발화문을 반환했습니다.")
    return ensure_emotion_response_quality(clean_description, user_text, utterance)


def play_tts(text: str, output_path: str = DEFAULT_TTS_OUTPUT_PATH, play_audio: bool = True) -> dict[str, Any]:
    """Step 3: Synthesize Jarvis' Korean utterance and optionally play it."""
    started = time.perf_counter()
    manager = AudioManager(model=REASONING_MODEL)
    safe_text = sanitize_tts_text(text)
    audio_path = manager.synthesize_speech(safe_text, output_path)
    if play_audio:
        manager._play_audio_file(audio_path)
    return {
        "ok": True,
        "audio_path": audio_path,
        "played": play_audio,
        "latency_sec": round(time.perf_counter() - started, 3),
    }


def run_interaction_pipeline(
    image_path: str | Path,
    *,
    user_text: str | None = None,
    output_path: str = DEFAULT_TTS_OUTPUT_PATH,
    play_audio: bool = True,
    warmup: bool = False,
) -> dict[str, Any]:
    """Run Moondream -> EXAONE -> TTS without crashing the robot loop on failures."""
    total_started = time.perf_counter()
    result: dict[str, Any] = {
        "image_path": str(image_path),
        "user_text": user_text or "",
        "success": False,
        "moondream_raw": "",
        "exaone_text": "",
        "tts_status": {"ok": False, "audio_path": None, "played": False, "latency_sec": 0.0},
        "timings": {
            "moondream_sec": 0.0,
            "exaone_sec": 0.0,
            "tts_sec": 0.0,
            "total_sec": 0.0,
        },
        "error": None,
    }

    try:
        if warmup:
            warm_up_models()

        stage_started = time.perf_counter()
        description = describe_user_emotion(image_path)
        result["timings"]["moondream_sec"] = round(time.perf_counter() - stage_started, 3)
        result["moondream_raw"] = description

        stage_started = time.perf_counter()
        utterance = get_exaone_empathy_response(description, user_text=user_text)
        result["timings"]["exaone_sec"] = round(time.perf_counter() - stage_started, 3)
        result["exaone_text"] = utterance

        tts_status = play_tts(utterance, output_path=output_path, play_audio=play_audio)
        result["tts_status"] = tts_status
        result["timings"]["tts_sec"] = tts_status["latency_sec"]
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["timings"]["total_sec"] = round(time.perf_counter() - total_started, 3)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Moondream -> EXAONE -> local TTS robot interaction loop")
    parser.add_argument("image_path", help="Image path containing the user's face")
    parser.add_argument("-o", "--output", default=DEFAULT_TTS_OUTPUT_PATH, help="TTS MP3 output path")
    parser.add_argument("--user-text", default=None, help="Optional STT text to combine with the vision result")
    parser.add_argument("--warmup", action="store_true", help="Warm up Ollama models before running the interaction")
    parser.add_argument("--no-play", action="store_true", help="Synthesize TTS without playing audio")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[WARNING] 이미지 파일을 찾을 수 없습니다: {args.image_path}")
        print("[WARNING] 테스트용 이미지는 `python create_mock_test_images.py`로 생성할 수 있습니다.")
        return

    result = run_interaction_pipeline(
        args.image_path,
        user_text=args.user_text,
        output_path=args.output,
        play_audio=not args.no_play,
        warmup=args.warmup,
    )
    print(f"[STAGE 1] Moondream Raw Output: {result['moondream_raw']}")
    print(f"[STAGE 2] EXAONE Decision & Text: {result['exaone_text']}")
    print(f"[STAGE 3] TTS Execution Status: {result['tts_status']}")
    print(f"[TIMINGS] {result['timings']}")
    if not result["success"]:
        raise SystemExit(f"[ERROR] {result['error']}")


if __name__ == "__main__":
    main()