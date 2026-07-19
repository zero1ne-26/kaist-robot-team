from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Deque

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_landmarker.task"
EMOTION_KEYS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
TRACKED_BLENDSHAPES = (
    "browDownLeft",
    "browDownRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "mouthPressLeft",
    "mouthPressRight",
    "noseSneerLeft",
    "noseSneerRight",
    "browInnerUp",
    "jawOpen",
    "eyeWideLeft",
    "eyeWideRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "cheekSquintLeft",
    "cheekSquintRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthPucker",
    "mouthShrugUpper",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
)


def check_model_file() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "face_landmarker.task 파일이 없습니다. "
            f"다음 경로에 모델 파일을 두세요: {MODEL_PATH}"
        )


def get_score(blendshape_dict: dict[str, float], name: str) -> float:
    return float(blendshape_dict.get(name, 0.0))


def _empty_emotion_scores(neutral: float = 1.0) -> dict[str, float]:
    return {
        "angry": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "happy": 0.0,
        "sad": 0.0,
        "surprise": 0.0,
        "neutral": neutral,
    }


def _normalize_active_blendshapes(blendshape_dict: dict[str, float]) -> dict[str, float]:
    active_sum = sum(max(0.0, get_score(blendshape_dict, name)) for name in TRACKED_BLENDSHAPES)
    if active_sum <= 1e-6:
        return {name: 0.0 for name in TRACKED_BLENDSHAPES}
    scale = min(1.0, 4.0 / active_sum)
    return {name: float(np.clip(get_score(blendshape_dict, name) * scale, 0.0, 1.0)) for name in TRACKED_BLENDSHAPES}


def _blendshape_variance(blendshape_dict: dict[str, float]) -> float:
    values = np.array([get_score(blendshape_dict, name) for name in TRACKED_BLENDSHAPES], dtype=np.float32)
    return float(np.var(values))


def _clip_scores(scores: dict[str, float]) -> dict[str, float]:
    return {key: float(np.clip(value, 0.0, 1.0)) for key, value in scores.items()}


def estimate_emotions(blendshape_dict: dict[str, float], noise_floor: float = 0.0008) -> dict[str, float]:
    """Logic-based ARKit blendshape emotion scoring.

    This function remains stateless for static images and tests. Real-time loops
    should use EmotionEngine, which smooths blendshapes before calling this logic.
    """
    if _blendshape_variance(blendshape_dict) < noise_floor:
        return _empty_emotion_scores(neutral=1.0)

    blendshape_dict = _normalize_active_blendshapes(blendshape_dict)
    brow_down = max(get_score(blendshape_dict, "browDownLeft"), get_score(blendshape_dict, "browDownRight"))
    eye_squint = max(get_score(blendshape_dict, "eyeSquintLeft"), get_score(blendshape_dict, "eyeSquintRight"))
    mouth_press = max(get_score(blendshape_dict, "mouthPressLeft"), get_score(blendshape_dict, "mouthPressRight"))
    nose_sneer = max(get_score(blendshape_dict, "noseSneerLeft"), get_score(blendshape_dict, "noseSneerRight"))
    brow_inner_up = get_score(blendshape_dict, "browInnerUp")
    jaw_open = get_score(blendshape_dict, "jawOpen")
    eye_wide = max(get_score(blendshape_dict, "eyeWideLeft"), get_score(blendshape_dict, "eyeWideRight"))
    mouth_smile = max(get_score(blendshape_dict, "mouthSmileLeft"), get_score(blendshape_dict, "mouthSmileRight"))
    cheek_squint = max(get_score(blendshape_dict, "cheekSquintLeft"), get_score(blendshape_dict, "cheekSquintRight"))
    mouth_frown = max(get_score(blendshape_dict, "mouthFrownLeft"), get_score(blendshape_dict, "mouthFrownRight"))
    mouth_pucker = get_score(blendshape_dict, "mouthPucker")
    mouth_shrug_upper = get_score(blendshape_dict, "mouthShrugUpper")
    mouth_lower_down = max(get_score(blendshape_dict, "mouthLowerDownLeft"), get_score(blendshape_dict, "mouthLowerDownRight"))
    mouth_dimple = max(get_score(blendshape_dict, "mouthDimpleLeft"), get_score(blendshape_dict, "mouthDimpleRight"))
    strong_genuine_smile = mouth_smile > 0.60 and (mouth_lower_down > 0.35 or cheek_squint > 0.20)
    genuine_open_smile = (mouth_smile > 0.35 and mouth_lower_down > 0.35 and jaw_open < 0.25) or strong_genuine_smile
    false_smile = (
        0.30 <= mouth_smile < 0.50
        and not genuine_open_smile
        and (eye_squint > 0.45 or nose_sneer > 0.30 or brow_down > 0.50)
    )
    shock_pattern = eye_wide > 0.20 and jaw_open > 0.20

    happy = (
        mouth_smile * 0.55
        + mouth_lower_down * 0.35
        + eye_squint * 0.10
        + cheek_squint * 0.10
        - jaw_open * 0.25
        - mouth_press * 0.10
    ) * 2.5
    if strong_genuine_smile:
        happy += 0.30
    if false_smile:
        happy *= 0.25

    angry = (
        jaw_open * 0.50
        + mouth_press * 0.25
        + mouth_dimple * 0.25
        + nose_sneer * 0.10
        + mouth_smile * 0.12
        - mouth_lower_down * 0.25
        - brow_inner_up * 0.15
    ) * 2.5
    if false_smile and (brow_down > 0.30 or eye_squint > 0.30):
        angry += 0.35

    sad = (
        mouth_frown * 0.50
        + jaw_open * 0.25
        + brow_down * 0.15
        + eye_squint * 0.10
        + brow_inner_up * 0.15
        - mouth_smile * 0.20
    ) * 2.5

    fear = (
        mouth_pucker * 0.45
        + brow_down * 0.25
        + eye_squint * 0.20
        + mouth_shrug_upper * 0.10
        + eye_wide * 0.15
        - jaw_open * 0.20
        - mouth_smile * 0.35
        - mouth_lower_down * 0.25
        - mouth_frown * 0.10
    ) * 3.0
    if shock_pattern:
        fear += 0.35

    surprise = (
        jaw_open * 0.45
        + eye_wide * 0.30
        + brow_inner_up * 0.25
        - mouth_smile * 0.20
    ) * 2.3
    if shock_pattern:
        surprise += 0.25

    disgust = (
        nose_sneer * 0.55
        + mouth_shrug_upper * 0.30
        + eye_squint * 0.15
        - mouth_smile * 0.10
    ) * 2.0
    if false_smile and nose_sneer > 0.20:
        disgust += 0.45

    emotion_scores = {
        "angry": angry,
        "disgust": disgust,
        "fear": fear,
        "happy": happy,
        "sad": sad,
        "surprise": surprise,
    }
    emotion_scores = _clip_scores(emotion_scores)

    if shock_pattern:
        emotion_scores["happy"] = min(emotion_scores["happy"], 0.15)
    if false_smile:
        emotion_scores["happy"] = min(emotion_scores["happy"], 0.25)
    if strong_genuine_smile and not shock_pattern:
        emotion_scores["angry"] = min(emotion_scores["angry"], emotion_scores["happy"] * 0.65)
        emotion_scores["fear"] = min(emotion_scores["fear"], emotion_scores["happy"] * 0.60)
        emotion_scores["sad"] = min(emotion_scores["sad"], emotion_scores["happy"] * 0.55)

    max_emotion_score = max(emotion_scores.values())
    neutral = max(0.0, 1.0 - max_emotion_score * 2.5)
    emotion_scores["neutral"] = float(np.clip(neutral, 0.0, 1.0))
    return emotion_scores


@dataclass
class EmotionEngine:
    """Stateful blendshape smoothing and logic-based emotion inference engine."""

    window_size: int = 10
    ema_alpha: float = 0.35
    noise_floor: float = 0.0008
    _history: Deque[dict[str, float]] = field(default_factory=deque, init=False)
    _ema: dict[str, float] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._history.clear()
        self._ema.clear()

    def update(self, blendshape_dict: dict[str, float]) -> dict[str, float]:
        normalized = _normalize_active_blendshapes(blendshape_dict)
        if not self._ema:
            self._ema = normalized.copy()
        else:
            for name in TRACKED_BLENDSHAPES:
                previous = self._ema.get(name, 0.0)
                current = normalized.get(name, 0.0)
                self._ema[name] = previous * (1.0 - self.ema_alpha) + current * self.ema_alpha

        self._history.append(self._ema.copy())
        while len(self._history) > self.window_size:
            self._history.popleft()

        smoothed = self.smoothed_blendshapes()
        return estimate_emotions(smoothed, noise_floor=self.noise_floor)

    def smoothed_blendshapes(self) -> dict[str, float]:
        if not self._history:
            return {name: 0.0 for name in TRACKED_BLENDSHAPES}
        return {
            name: float(np.mean([frame.get(name, 0.0) for frame in self._history]))
            for name in TRACKED_BLENDSHAPES
        }


def draw_ai_view(face_landmarks, emotion_scores, source_width: int, source_height: int, width: int = 720, height: int = 480):
    ai_view = np.zeros((height, width, 3), dtype=np.uint8)
    if face_landmarks is not None:
        points = np.array([[lm.x * source_width, lm.y * source_height] for lm in face_landmarks])
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        face_w, face_h = max_x - min_x, max_y - min_y
        if face_w > 0 and face_h > 0:
            face_area_x, face_area_y, face_area_w, face_area_h = 80, 70, 280, 320
            scale = min(face_area_w / face_w, face_area_h / face_h)
            offset_x = face_area_x + (face_area_w - face_w * scale) / 2
            offset_y = face_area_y + (face_area_h - face_h * scale) / 2
            for x, y in points:
                draw_x = int((x - min_x) * scale + offset_x)
                draw_y = int((y - min_y) * scale + offset_y)
                if 0 <= draw_x < width and 0 <= draw_y < height:
                    cv2.circle(ai_view, (draw_x, draw_y), 2, (0, 255, 0), -1)

    emotion_order = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    start_x, start_y, bar_w, gap = 470, 70, 130, 32
    for index, emotion in enumerate(emotion_order):
        score = emotion_scores.get(emotion, 0.0)
        y = start_y + index * gap
        cv2.putText(ai_view, f"{emotion}: {score:.2f}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.rectangle(ai_view, (start_x + 120, y - 14), (start_x + 120 + bar_w, y), (80, 80, 80), 1)
        cv2.rectangle(ai_view, (start_x + 120, y - 14), (start_x + 120 + int(bar_w * score), y), (0, 255, 0), -1)
    return ai_view


def create_face_landmarker(running_mode):
    check_model_file()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def emotion_scores_from_result(result, engine: EmotionEngine | None = None) -> tuple[dict[str, float], object | None]:
    default_scores = _empty_emotion_scores(neutral=1.0)
    if not result.face_landmarks or not result.face_blendshapes:
        if engine is not None:
            engine.reset()
        return default_scores, None
    blendshape_dict = {category.category_name: float(category.score) for category in result.face_blendshapes[0]}
    scores = engine.update(blendshape_dict) if engine is not None else estimate_emotions(blendshape_dict)
    return scores, result.face_landmarks[0]