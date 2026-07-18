from __future__ import annotations

import gc
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from audio_manager import AudioManager


class PipelineState(Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    INTERRUPTED = "INTERRUPTED"


@dataclass
class FastLoopSnapshot:
    presence: bool = False
    face_expression: str = "neutral"
    gaze_direction: str = "center"
    upper_body_posture: str = "stable"
    source: str = "none"


@dataclass
class PipelineResult:
    transcribed_text: str = ""
    llm_reply: str = ""
    audio_paths: list[str] = None  # type: ignore[assignment]
    audio_path: Path | str | None = None
    state: str = PipelineState.RUNNING.value
    interrupted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "transcribed_text": self.transcribed_text,
            "llm_reply": self.llm_reply,
            "audio_paths": self.audio_paths or [],
            "audio_path": self.audio_path,
            "state": self.state,
            "interrupted": self.interrupted,
        }


@dataclass
class ConversationTurn:
    user_text: str
    assistant_text: str
    timestamp: float


class DualLoopTriggerGate:
    """Visual trigger와 transcript를 바탕으로 slow loop 시작 여부를 판단합니다."""

    def __init__(self, expression_threshold: int = 3) -> None:
        self.expression_threshold = expression_threshold
        self._frown_count = 0

    def should_trigger_on_visual(self, snapshot: FastLoopSnapshot) -> bool:
        if snapshot.presence:
            return True
        if snapshot.face_expression in {"frown", "sad", "angry"}:
            self._frown_count += 1
        else:
            self._frown_count = 0
        return self._frown_count >= self.expression_threshold

    def should_invoke_slow_loop(self, transcript: str, snapshot: FastLoopSnapshot) -> bool:
        if transcript.strip():
            return True
        return self.should_trigger_on_visual(snapshot)


def _snapshot_from_mapping(payload: Dict[str, Any]) -> FastLoopSnapshot:
    return FastLoopSnapshot(
        presence=bool(payload.get("presence", False)),
        face_expression=str(payload.get("face_expression", "neutral")),
        gaze_direction=str(payload.get("gaze_direction", "center")),
        upper_body_posture=str(payload.get("upper_body_posture", "stable")),
        source=str(payload.get("source", "none")),
    )


def _safe_queue_put(target_queue: Any, item: Dict[str, Any]) -> None:
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass


def _start_memory_monitor(role: str, interval: float, stop_event: Any) -> Optional[threading.Thread]:
    return None


def _build_audio_manager(runtime_config: Dict[str, Any]) -> AudioManager:
    return AudioManager(
        base_url=runtime_config.get("base_url"),
        api_key=runtime_config.get("api_key", "dummy"),
        model=runtime_config.get("model", "exaone3.5:2.4b"),
        system_prompt=runtime_config.get("system_prompt"),
    )


def _fast_loop_worker(
    runtime_config: Dict[str, Any],
    event_queue: Any,
    stop_event: Any,
    mic_timeout: float,
    mic_phrase_time_limit: float,
    fast_loop_interval: float,
    memory_profile_interval: float,
) -> None:
    audio_manager = _build_audio_manager(runtime_config)

    try:
        try:
            initial_visual = audio_manager.get_visual_context()
            _safe_queue_put(
                event_queue,
                {"kind": "visual", "payload": initial_visual, "timestamp": time.monotonic()},
            )
        except Exception as exc:
            _safe_queue_put(event_queue, {"kind": "error", "payload": str(exc), "stage": "visual"})

        transcript = audio_manager.transcribe(
            runtime_config.get("input_type", "mic"),
            runtime_config.get("file_path"),
            "ko",
            mic_timeout,
            mic_phrase_time_limit,
        )
        _safe_queue_put(
            event_queue,
            {"kind": "transcript", "payload": transcript, "timestamp": time.monotonic()},
        )

        while not stop_event.is_set():
            try:
                visual_context = audio_manager.get_visual_context()
                _safe_queue_put(
                    event_queue,
                    {"kind": "visual", "payload": visual_context, "timestamp": time.monotonic()},
                )
            except Exception as exc:
                _safe_queue_put(event_queue, {"kind": "error", "payload": str(exc), "stage": "visual"})
                break

            if stop_event.wait(fast_loop_interval):
                break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        audio_manager.close()
        gc.collect()


def _slow_loop_worker(
    runtime_config: Dict[str, Any],
    event_queue: Any,
    result_queue: Any,
    stop_event: Any,
    memory_profile_interval: float,
) -> None:
    audio_manager = _build_audio_manager(runtime_config)
    latest_visual = FastLoopSnapshot()
    gate = DualLoopTriggerGate()
    turn_cleaned = False

    try:
        while not stop_event.is_set():
            try:
                event = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            kind = event.get("kind")
            payload = event.get("payload")

            if kind == "error":
                result_queue.put(
                    {
                        "transcribed_text": "",
                        "llm_reply": "",
                        "audio_paths": [],
                        "audio_path": runtime_config.get("output_path"),
                        "state": PipelineState.FAILED.value,
                        "error": payload,
                    }
                )
                stop_event.set()
                break

            if kind == "visual" and isinstance(payload, dict):
                latest_visual = _snapshot_from_mapping(payload)
                continue

            if kind != "transcript":
                continue

            transcript = str(payload or "")
            lowered_transcript = transcript.lower()
            if any(keyword in lowered_transcript for keyword in ("종료", "그만", "quit", "exit", "stop")):
                print("[SYSTEM] 종료 키워드를 감지해 안전 종료합니다.", flush=True)
                result_queue.put(
                    {
                        "transcribed_text": transcript,
                        "llm_reply": "",
                        "audio_paths": [],
                        "audio_path": runtime_config.get("output_path"),
                        "state": PipelineState.STOPPED.value,
                    }
                )
                stop_event.set()
                break

            if not gate.should_invoke_slow_loop(transcript, latest_visual):
                print("[FAST LOOP] 임계치 미충족: slow loop 호출 생략", flush=True)
                result_queue.put(
                    {
                        "transcribed_text": transcript,
                        "llm_reply": "",
                        "audio_paths": [],
                        "audio_path": runtime_config.get("output_path"),
                        "state": PipelineState.COMPLETED.value,
                    }
                )
                stop_event.set()
                break

            audio_manager.transcribed_text = transcript
            audio_manager.run_memory_maintenance("stt_to_llm")
            print(f"[STT 결과] : {transcript}", flush=True)

            output_path = Path(runtime_config.get("output_path", "response.mp3"))
            audio_manager.start_tts_worker(output_path.parent, output_path.stem)
            context_prompt = (
                f"[voice]\n{transcript}\n\n"
                f"[visual_context]\n"
                f"presence={latest_visual.presence}, face_expression={latest_visual.face_expression}, "
                f"gaze={latest_visual.gaze_direction}, posture={latest_visual.upper_body_posture}, source={latest_visual.source}"
            )
            llm_sentences: list[str] = []
            for sentence in audio_manager.stream_remote_llm(context_prompt):
                llm_sentences.append(sentence)
                audio_manager.enqueue_sentence(sentence)

            if not llm_sentences:
                print("[STATE] 경고: LLM 스트리밍에서 생성된 문장이 없습니다.", flush=True)

            llm_reply = " ".join(llm_sentences).strip()
            print(f"[LLM 판단] : {llm_reply}", flush=True)

            audio_manager.run_memory_maintenance("llm_to_tts_finalize")
            audio_manager.stop_tts_worker()
            audio_manager.run_memory_maintenance("tts_to_idle")
            result_queue.put(
                {
                    "transcribed_text": transcript,
                    "llm_reply": llm_reply,
                    "audio_paths": audio_manager.result_paths,
                    "audio_path": audio_manager.result_paths[0] if audio_manager.result_paths else output_path,
                    "state": PipelineState.COMPLETED.value,
                }
            )

            del llm_sentences
            del context_prompt
            turn_cleaned = True
            gc.collect()
            stop_event.set()
            break
    except KeyboardInterrupt:
        stop_event.set()
        result_queue.put(
            {
                "transcribed_text": getattr(audio_manager, "transcribed_text", ""),
                "llm_reply": "",
                "audio_paths": getattr(audio_manager, "result_paths", []),
                "audio_path": runtime_config.get("output_path"),
                "state": PipelineState.STOPPED.value,
            }
        )
    except Exception as exc:
        gc.collect()
        result_queue.put(
            {
                "transcribed_text": getattr(audio_manager, "transcribed_text", ""),
                "llm_reply": "",
                "audio_paths": getattr(audio_manager, "result_paths", []),
                "audio_path": runtime_config.get("output_path"),
                "state": PipelineState.FAILED.value,
                "error": str(exc),
            }
        )
        stop_event.set()
    finally:
        audio_manager.close()
        if not turn_cleaned:
            gc.collect()


class ConversationStateMachine:
    """멀티프로세싱 기반 STT -> LLM -> TTS 데몬 슈퍼바이저."""

    def __init__(
        self,
        audio_manager: Any | None = None,
        output_path: str | Path = "response.mp3",
        input_type: str = "mic",
        file_path: Optional[str] = None,
        mic_timeout: float = 10.0,
        mic_phrase_time_limit: float = 10.0,
        fast_loop_interval: float = 0.2,
        context_window_size: int = 16,
        memory_profile_interval: float = 15.0,
        shutdown_grace_period: float = 5.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.input_type = input_type
        self.file_path = file_path
        self.mic_timeout = mic_timeout
        self.mic_phrase_time_limit = mic_phrase_time_limit
        self.fast_loop_interval = fast_loop_interval
        self.context_window_size = context_window_size
        self.memory_profile_interval = memory_profile_interval
        self.shutdown_grace_period = shutdown_grace_period

        self.base_url = getattr(audio_manager, "base_url", None)
        self.api_key = getattr(audio_manager, "api_key", "dummy") if audio_manager is not None else "dummy"
        self.model = getattr(audio_manager, "model", "exaone3.5:2.4b") if audio_manager is not None else "exaone3.5:2.4b"
        self.system_prompt = getattr(audio_manager, "system_prompt", None) if audio_manager is not None else None

        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._event_queue = self._ctx.Queue(maxsize=max(16, context_window_size * 2))
        self._result_queue = self._ctx.Queue(maxsize=4)
        self._processes: list[mp.Process] = []
        self._parent_monitor_stop = threading.Event()
        self._parent_monitor_thread: Optional[threading.Thread] = None
        self.current_state = PipelineState.RUNNING

    def _runtime_config(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "input_type": self.input_type,
            "file_path": self.file_path,
            "output_path": str(self.output_path),
        }

    def _start_parent_memory_monitor(self) -> None:
        return

    def shutdown(self) -> None:
        self.current_state = PipelineState.STOPPED
        self._stop_event.set()
        self._parent_monitor_stop.set()

    def _start_processes(self) -> None:
        runtime_config = self._runtime_config()
        fast_process = self._ctx.Process(
            target=_fast_loop_worker,
            name="fast-loop",
            args=(
                runtime_config,
                self._event_queue,
                self._stop_event,
                self.mic_timeout,
                self.mic_phrase_time_limit,
                self.fast_loop_interval,
                self.memory_profile_interval,
            ),
        )
        slow_process = self._ctx.Process(
            target=_slow_loop_worker,
            name="slow-loop",
            args=(
                runtime_config,
                self._event_queue,
                self._result_queue,
                self._stop_event,
                self.memory_profile_interval,
            ),
        )
        fast_process.start()
        slow_process.start()
        self._processes = [fast_process, slow_process]

    def _cleanup_processes(self) -> None:
        self.shutdown()
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=self.shutdown_grace_period)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=1)
        self._parent_monitor_stop.set()
        if self._parent_monitor_thread is not None:
            self._parent_monitor_thread.join(timeout=1)

    def run(self) -> Dict[str, Any]:
        self._start_parent_memory_monitor()
        self._start_processes()
        result: Dict[str, Any] = {
            "transcribed_text": "",
            "llm_reply": "",
            "audio_paths": [],
            "audio_path": self.output_path,
            "state": PipelineState.RUNNING.value,
        }

        try:
            while True:
                if self._stop_event.is_set() and self._result_queue.empty():
                    break
                try:
                    result = self._result_queue.get(timeout=0.25)
                    break
                except queue.Empty:
                    if not any(process.is_alive() for process in self._processes):
                        break
        finally:
            self._cleanup_processes()

        self.current_state = PipelineState(result.get("state", PipelineState.STOPPED.value)) if result.get("state") in PipelineState._value2member_map_ else PipelineState.STOPPED
        gc.collect()
        return result
