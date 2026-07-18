from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import requests


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "moondream")
VLM_TIMEOUT = float(os.getenv("OLLAMA_VLM_TIMEOUT_SECONDS", "60"))
VLM_IMAGE_SIZE = int(os.getenv("VLM_IMAGE_SIZE", "384"))
VLM_NUM_CTX = int(os.getenv("VLM_NUM_CTX", "1024"))
VLM_NUM_PREDICT = int(os.getenv("VLM_NUM_PREDICT", "128"))

BASELINE_SEC = float(os.getenv("FACE_BASELINE_SEC", "2.0"))
STRIDE = int(os.getenv("VLM_STRIDE", "5"))
SW_THRESH = float(os.getenv("VLM_SW_THRESH", "0.03"))
SY_THRESH = float(os.getenv("VLM_SY_THRESH", "0.05"))
VLM_COOLDOWN = float(os.getenv("VLM_COOLDOWN", "2.0"))
SNAPSHOT_DIR = Path(os.getenv("VLM_SNAPSHOT_DIR", "./events"))
CROP_PADDING = float(os.getenv("VLM_CROP_PADDING", "0.3"))
CROP_SIZE = int(os.getenv("VLM_CROP_SIZE", "640"))


def _require_model_file() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "face_landmarker.task 파일이 없습니다. "
            f"다음 경로에 모델 파일을 두세요: {MODEL_PATH}"
        )


def build_landmarker():
    _require_model_file()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def blendshape_dict(blendshapes: Any) -> dict[str, float]:
    return {category.category_name: float(category.score) for category in blendshapes}


def _score(scores: dict[str, float], name: str) -> float:
    return float(scores.get(name, 0.0))


def classify(delta: dict[str, float]) -> tuple[str, str]:
    smile = max(_score(delta, "mouthSmileLeft"), _score(delta, "mouthSmileRight"))
    cheek = max(_score(delta, "cheekSquintLeft"), _score(delta, "cheekSquintRight"))
    jaw = _score(delta, "jawOpen")
    eye_wide = max(_score(delta, "eyeWideLeft"), _score(delta, "eyeWideRight"))
    brow_inner = _score(delta, "browInnerUp")
    brow_down = max(_score(delta, "browDownLeft"), _score(delta, "browDownRight"))
    eye_squint = max(_score(delta, "eyeSquintLeft"), _score(delta, "eyeSquintRight"))
    mouth_press = max(_score(delta, "mouthPressLeft"), _score(delta, "mouthPressRight"))
    mouth_frown = max(_score(delta, "mouthFrownLeft"), _score(delta, "mouthFrownRight"))

    candidates = {
        "웃음": smile * 0.7 + cheek * 0.3,
        "놀람": jaw * 0.45 + eye_wide * 0.35 + brow_inner * 0.2,
        "찡그림": brow_down * 0.45 + eye_squint * 0.25 + mouth_press * 0.2 + mouth_frown * 0.1,
    }
    label, value = max(candidates.items(), key=lambda item: item[1])
    if value < 0.08:
        return "무표정", "기저선 대비 변화가 작음"
    return label, f"기저선 대비 {label} 점수 {value:.3f}"


def encode_image(path: str | Path, max_size: int = VLM_IMAGE_SIZE) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")

    height, width = image.shape[:2]
    scale = min(max_size / max(width, height), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError(f"JPEG 인코딩 실패: {path}")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_scene_result(payload: dict[str, Any], raw_text: str = "") -> dict[str, Any]:
    action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
    situation = str(payload.get("situation") or "").strip()
    if not situation or situation == "지금 상황을 한국어 한 문장으로":
        situation = _fallback_situation(raw_text)
    normalized = {
        "situation": situation,
        "user_state": str(payload.get("user_state") or "주의필요"),
        "perception_conflict": bool(payload.get("perception_conflict", False)),
        "should_speak": bool(payload.get("should_speak", False)),
        "utterance": payload.get("utterance"),
        "action": {
            "device": str(action.get("device") or "none"),
            "command": str(action.get("command") or "none"),
            "reason": str(action.get("reason") or ""),
        },
        "raw_text": raw_text,
    }
    if not normalized["should_speak"]:
        normalized["utterance"] = None
    if normalized["action"]["device"] not in {"light", "plug", "aircon", "none"}:
        normalized["action"]["device"] = "none"
    if normalized["action"]["command"] not in {"on", "off", "dim", "none"}:
        normalized["action"]["command"] = "none"
    return normalized


def _fallback_situation(raw_text: str) -> str:
    compact = re.sub(r"\s+", " ", raw_text or "").strip()
    if not compact:
        return "이미지를 확인했지만 상황 설명을 만들지 못했습니다."
    compact = re.sub(r"[{}\[\]\"]", "", compact)
    return compact[:80] or "이미지를 확인했습니다."


def describe_scene(image_path: str | Path, event_info: str, command: str | None = None) -> dict[str, Any]:
    prompt = f"""너는 '자비스'라는 로컬 VLM 비서다. 카메라 이미지와 빠른 인지 루프 관측을 함께 보고 판단한다.

[빠른 인지 루프 관측]
{event_info or "없음"}

이 관측은 참고 정보다. 이미지와 다르면 이미지를 더 믿어라.

[사용자 발화]
{command if command else "없음"}

[제어 가능한 가상 IoT 기기]
- light: on / off / dim
- plug: on / off
- aircon: on / off

JSON 객체 하나로만 답하라. 설명, markdown, 코드블록은 금지한다.
{{
  "situation": "지금 상황을 한국어 한 문장으로",
  "user_state": "집중" | "휴식" | "부재" | "주의필요",
  "perception_conflict": true 또는 false,
  "should_speak": true 또는 false,
  "utterance": "사용자에게 할 말" 또는 null,
  "action": {{
    "device": "light" | "plug" | "aircon" | "none",
    "command": "on" | "off" | "dim" | "none",
    "reason": "판단 근거"
  }}
}}

판단 규칙:
1. 기본은 침묵이다. 말을 거는 것은 사용자를 방해할 수 있다.
2. should_speak는 사용자가 말을 걸었거나, 놀람/찡그림/고통/위험처럼 주의가 필요할 때만 true다.
3. should_speak가 false면 utterance는 반드시 null이다.
4. 사용자 발화가 없으면 기기를 함부로 조작하지 마라. 명백한 부재로 조명을 끄는 경우 외에는 device는 "none"이다.
5. perception_conflict는 빠른 루프의 표정 판정과 이미지 속 실제 표정이 다를 때만 true다.
"""
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encode_image(image_path)],
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": VLM_NUM_PREDICT,
            "num_ctx": VLM_NUM_CTX,
        },
    }
    started = time.time()
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=VLM_TIMEOUT)
    response.raise_for_status()
    content = str(response.json().get("message", {}).get("content", ""))
    parsed = _extract_json_object(content)
    result = _normalize_scene_result(parsed, content)
    result["latency_sec"] = round(time.time() - started, 3)
    result["model"] = VLM_MODEL
    return result


def extract(pose: Any, frame: np.ndarray):
    small = cv2.resize(frame, (640, 360))
    result = pose.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return small, None, None
    landmarks = result.pose_landmarks.landmark
    shoulder_width = abs(landmarks[11].x - landmarks[12].x)
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    return small, (shoulder_width, shoulder_y), landmarks


def crop_person(frame: np.ndarray, landmarks: Any, padding: float = CROP_PADDING) -> np.ndarray:
    height, width = frame.shape[:2]
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(width, int(x_max + pad_x))
    y2 = min(height, int(y_max + pad_y))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame

    crop_h, crop_w = crop.shape[:2]
    scale = min(CROP_SIZE / crop_w, CROP_SIZE / crop_h, 1.0)
    new_w, new_h = int(round(crop_w * scale)), int(round(crop_h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (CROP_SIZE - new_w) // 2
    pad_right = CROP_SIZE - new_w - pad_left
    pad_top = (CROP_SIZE - new_h) // 2
    pad_bottom = CROP_SIZE - new_h - pad_top
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def classify_event(prev: tuple[float, float] | None, cur: tuple[float, float] | None) -> str | None:
    present, was_present = cur is not None, prev is not None
    if present and not was_present:
        return "등장"
    if not present and was_present:
        return "퇴장"
    if present and was_present:
        if abs(cur[0] - prev[0]) > SW_THRESH:
            return "몸 방향 전환"
        if abs(cur[1] - prev[1]) > SY_THRESH:
            return "자세 변화"
    return None


def analyze_face(landmarker: Any, frame: np.ndarray, idx: int, fps: float, state: dict[str, Any]):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(idx / fps * 1000)
    sec = idx / fps

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    detected = bool(result.face_landmarks)

    if sec < BASELINE_SEC:
        if detected and result.face_blendshapes:
            scores = blendshape_dict(result.face_blendshapes[0])
            for key, value in scores.items():
                state["baseline_sum"][key] = state["baseline_sum"].get(key, 0.0) + value
            state["baseline_count"] += 1
        return "기저선", None

    if state["baseline"] is None:
        if state["baseline_count"] > 0:
            state["baseline"] = {
                key: value / state["baseline_count"]
                for key, value in state["baseline_sum"].items()
            }
            print(f"[기저선 확정: {state['baseline_count']}개 샘플 평균 사용]\n")
        else:
            state["baseline"] = {}
            print("[경고: 기저선 구간에 얼굴 미검출 - 기저선을 0으로 사용]\n")

    if not detected or not result.face_blendshapes:
        return "얼굴X", None

    scores = blendshape_dict(result.face_blendshapes[0])
    baseline = state["baseline"] or {}
    delta = {key: scores.get(key, 0.0) - baseline.get(key, 0.0) for key in scores}
    label, _reason = classify(delta)

    face_change = None
    if state["prev_label"] is not None and label != state["prev_label"]:
        face_change = (state["prev_label"], label)
    state["prev_label"] = label
    return label, face_change


def build_event_info(pose_event: str | None, face_change: tuple[str, str] | None) -> str:
    sentences: list[str] = []
    if pose_event == "등장":
        sentences.append("사용자가 화면에 등장함.")
    elif pose_event == "퇴장":
        sentences.append("사용자가 화면에서 사라짐.")
    elif pose_event in {"자세 변화", "몸 방향 전환"}:
        sentences.append(f"사용자의 {pose_event}이 감지됨.")

    if face_change:
        prev_label, cur_label = face_change
        sentences.append(f"표정이 {prev_label}에서 {cur_label}으로 바뀜.")
    return " ".join(sentences)


def run(path: str = "scene.mp4") -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {path}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = build_landmarker()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    total_frames = 0
    fast_runs = 0
    vlm_calls = 0
    face_events = 0
    last_vlm_t = -1e9
    prev = None
    idx = 0
    face_time = 0.0
    pose_time = 0.0
    face_state = {
        "baseline_sum": {},
        "baseline_count": 0,
        "baseline": None,
        "prev_label": None,
    }

    print(f"[VLM 모델: {VLM_MODEL} | host: {OLLAMA_HOST}]")
    print(f"[기저선 구간: 0.00s ~ {BASELINE_SEC:.2f}s 동안 blendshape 평균 계산]\n")
    started = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            total_frames += 1

            if idx % STRIDE != 0:
                idx += 1
                continue

            fast_runs += 1
            now = time.time()
            sec = idx / fps

            pose_started = time.time()
            small, cur, landmarks = extract(pose, frame)
            pose_time += time.time() - pose_started

            face_started = time.time()
            _label, face_change = analyze_face(landmarker, frame, idx, fps, face_state)
            face_time += time.time() - face_started

            pose_event = classify_event(prev, cur)
            present = cur is not None
            events: list[str] = []
            if pose_event:
                events.append(pose_event)
            if face_change:
                events.append(f"표정 변화({face_change[0]}->{face_change[1]})")
                face_events += 1
            event = " + ".join(events) if events else None

            if event and (now - last_vlm_t) >= VLM_COOLDOWN:
                full_path = SNAPSHOT_DIR / f"full_{idx:05d}.jpg"
                cv2.imwrite(str(full_path), frame)
                if present and landmarks is not None:
                    event_path = SNAPSHOT_DIR / f"event_{idx:05d}.jpg"
                    cv2.imwrite(str(event_path), crop_person(small, landmarks))

                event_info = build_event_info(pose_event, face_change)
                print(f"[{sec:7.2f}s] {event}")
                print(f"           event_info: {event_info}")
                try:
                    vlm_started = time.time()
                    result = describe_scene(full_path, event_info, command=None)
                    vlm_dt = time.time() - vlm_started
                    vlm_calls += 1
                    last_vlm_t = now
                    action = result.get("action") or {}
                    print(
                        f"           VLM({vlm_dt:.1f}s): "
                        f"should_speak={result.get('should_speak')} "
                        f"user_state={result.get('user_state')} "
                        f"perception_conflict={result.get('perception_conflict')} "
                        f"action={action.get('device')}/{action.get('command')}"
                    )
                    if result.get("should_speak"):
                        print(f"           >>> 자비스: \"{result.get('utterance')}\"")
                except Exception as exc:
                    print(f"           VLM 실패: {type(exc).__name__}: {exc}")
            elif event:
                event_info = build_event_info(pose_event, face_change)
                print(f"[{sec:7.2f}s] {event} (VLM 생략) | {event_info}")

            prev = cur
            idx += 1
    finally:
        capture.release()
        pose.close()
        landmarker.close()

    elapsed = time.time() - started
    ratio = total_frames / vlm_calls if vlm_calls else float("inf")
    avg_face = face_time / fast_runs if fast_runs else 0.0
    avg_pose = pose_time / fast_runs if fast_runs else 0.0
    avg_frame = elapsed / total_frames if total_frames else 0.0

    print()
    print(f"전체 프레임 수: {total_frames}")
    print(f"빠른 루프 실행 횟수: {fast_runs}")
    print(f"VLM 호출 횟수: {vlm_calls}")
    print(f"표정 변화 이벤트 횟수: {face_events}")
    print(f"절감 배수: {ratio:.1f}x" if vlm_calls else "절감 배수: N/A")
    print(f"FaceLandmarker 평균 처리 시간: {avg_face * 1000:.1f}ms")
    print(f"Pose 평균 처리 시간: {avg_pose * 1000:.1f}ms")
    print(f"전체 프레임당 평균 처리 시간: {avg_frame * 1000:.1f}ms")
    print(f"총 처리 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "scene.mp4")