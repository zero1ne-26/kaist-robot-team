from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_mediapipe_emotion() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "mediapipe-llm-tts" / "mediapipe_emotion.py"
    spec = importlib.util.spec_from_file_location("mediapipe_emotion", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"mediapipe_emotion 모듈을 로드할 수 없습니다: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mediapipe_emotion = _load_mediapipe_emotion()

EmotionEngine = _mediapipe_emotion.EmotionEngine
create_face_landmarker = _mediapipe_emotion.create_face_landmarker
draw_ai_view = _mediapipe_emotion.draw_ai_view
emotion_scores_from_result = _mediapipe_emotion.emotion_scores_from_result
