from __future__ import annotations

import time
from typing import Any

from utils.audio_manager import AudioManager
from utils.loop4 import REASONING_MODEL, sanitize_tts_text


def speak_tts(text: str, output_path: str = "interactive_response.mp3", play_audio: bool = True) -> dict[str, Any]:
    """Synthesize sanitized Korean text and optionally play it with the local audio backend."""
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
