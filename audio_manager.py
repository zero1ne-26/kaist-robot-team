import os
import io
import gc
import queue
import re
import json
import base64
import logging
import subprocess
import threading
import time
from collections import deque
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from gtts import gTTS
import requests

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency
    ollama = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency
    PeftModel = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional dependency
    sr = None

try:
    from paho.mqtt import publish as mqtt_publish
except ImportError:  # pragma: no cover - optional dependency
    mqtt_publish = None


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


_AVATAR_STATE: Dict[str, str] = {
    "emotion": "neutral",
    "gesture": "idle",
}


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0

    def allow_request(self) -> bool:
        if self.state == "OPEN":
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ConversationMemoryBuffer:
    """최근 대화 턴만 유지하는 경량 메모리 버퍼."""

    def __init__(self, max_turns: int = 6, max_chars: int = 1200) -> None:
        self.max_turns = max(1, max_turns)
        self.max_chars = max(120, max_chars)
        self._turns: Deque[Dict[str, str]] = deque(maxlen=self.max_turns)
        self._lock = threading.Lock()

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user = re.sub(r"\s+", " ", user_text or "").strip()
        assistant = re.sub(r"\s+", " ", assistant_text or "").strip()
        if not user and not assistant:
            return
        with self._lock:
            self._turns.append({"user": user, "assistant": assistant})

    def render(self) -> str:
        with self._lock:
            turns = list(self._turns)
        rendered: List[str] = []
        budget = self.max_chars
        for turn in reversed(turns):
            line = f"사용자: {turn.get('user', '')}\n로봇: {turn.get('assistant', '')}".strip()
            if not line:
                continue
            if len(line) > budget:
                line = line[:budget]
            rendered.append(line)
            budget -= len(line)
            if budget <= 0:
                break
        return "\n".join(reversed(rendered))

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()


def get_visual_context() -> Dict[str, Any]:
    """인지 모듈(fast loop)에서 시각 맥락 정보를 가져오는 스캐폴딩 함수."""
    return {
        "presence": False,
        "face_expression": "neutral",
        "gaze_direction": "center",
        "upper_body_posture": "stable",
        "source": "realsense_fast_loop_stub",
    }


def _translate_weather_description(description: str) -> str:
    normalized = (description or "").strip().lower()
    if not normalized:
        return "정보 없음"

    mappings = {
        "clear": "맑음",
        "sunny": "맑음",
        "partly cloudy": "구름 조금",
        "cloudy": "흐림",
        "overcast": "구름 많음",
        "light rain": "약한 비",
        "moderate rain": "비",
        "heavy rain": "강한 비",
        "patchy rain possible": "가끔 비",
        "drizzle": "이슬비",
        "mist": "옅은 안개",
        "fog": "안개",
        "thunderstorm": "뇌우",
        "snow": "눈",
    }
    for key, value in mappings.items():
        if key in normalized:
            return value

    return description


def get_current_weather(location: str = "Daejeon") -> Dict[str, Any]:
    """wttr.in JSON API로 현재 날씨를 조회해 딕셔너리로 반환합니다."""
    safe_location = (location or "Daejeon").strip() or "Daejeon"

    url = f"https://wttr.in/{safe_location}?format=j1"
    try:
        response = requests.get(
            url,
            timeout=3.0,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        print(f"[DEBUG] 날씨 API 에러: {e}", flush=True)
        return {
            "error": "timeout",
            "message": "기상청 서버 응답 지연",
            "location": safe_location,
        }
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] 날씨 API 에러: {e}", flush=True)
        return {
            "error": "request_error",
            "message": "기상 통신망 요청 실패",
            "location": safe_location,
        }

    try:
        payload = response.json()
    except ValueError as e:
        print(f"[DEBUG] 날씨 API 에러: {e}", flush=True)
        return {
            "error": "invalid_json",
            "message": "기상 데이터 JSON 파싱 실패",
            "location": safe_location,
        }

    current_condition = (payload.get("current_condition") or [{}])[0] if isinstance(payload, dict) else {}
    nearest_area = (payload.get("nearest_area") or [{}])[0] if isinstance(payload, dict) else {}
    area_name_obj = (nearest_area.get("areaName") or [{}])[0] if isinstance(nearest_area, dict) else {}
    weather_desc_obj = (current_condition.get("weatherDesc") or [{}])[0] if isinstance(current_condition, dict) else {}
    weather_description = weather_desc_obj.get("value") or ""
    weather_ko = _translate_weather_description(weather_description)
    temperature_c = current_condition.get("temp_C")

    return {
        "location": area_name_obj.get("value") or safe_location,
        "temperature_c": temperature_c,
        "weather": weather_description,
        "weather_ko": weather_ko,
        "summary_ko": f"현재 {area_name_obj.get('value') or safe_location}의 날씨는 {weather_ko}이고 기온은 {temperature_c}도씨입니다.",
        "raw": payload,
    }


def get_weather(location: str = "Daejeon") -> Dict[str, Any]:
    """로봇 음성 응답용 날씨 도구. 3초 안에 실패하면 명시적 에러를 반환합니다."""
    safe_location = (location or "Daejeon").strip() or "Daejeon"
    try:
        response = requests.get(
            f"https://wttr.in/{safe_location}?format=j1",
            timeout=3.0,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"[DEBUG] 날씨 API 에러: {exc}", flush=True)
        return {
            "error": "weather_service_timeout",
            "message": "날씨 정보를 불러오는 데 실패했습니다. 통신 상태를 확인해주세요.",
            "location": safe_location,
        }

    current_condition = (payload.get("current_condition") or [{}])[0] if isinstance(payload, dict) else {}
    nearest_area = (payload.get("nearest_area") or [{}])[0] if isinstance(payload, dict) else {}
    area_name_obj = (nearest_area.get("areaName") or [{}])[0] if isinstance(nearest_area, dict) else {}
    weather_desc_obj = (current_condition.get("weatherDesc") or [{}])[0] if isinstance(current_condition, dict) else {}
    weather_description = weather_desc_obj.get("value") or ""
    weather_ko = _translate_weather_description(weather_description)
    temperature_c = current_condition.get("temp_C")

    return {
        "status": "ok",
        "location": area_name_obj.get("value") or safe_location,
        "temperature_c": temperature_c,
        "weather": weather_description,
        "weather_ko": weather_ko,
        "summary_ko": f"현재 {area_name_obj.get('value') or safe_location}의 날씨는 {weather_ko}이고 기온은 {temperature_c}도씨입니다.",
        "raw": payload,
    }


def search_web(query: str) -> Dict[str, Any]:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return {"error": "invalid_query", "message": "query가 필요합니다."}

    return {
        "status": "ok",
        "query": normalized_query,
        "result": f"웹 검색 결과: {normalized_query}",
    }


def set_alarm(time_value: str) -> Dict[str, Any]:
    normalized_time = (time_value or "").strip()
    if not normalized_time:
        return {"error": "invalid_time", "message": "time이 필요합니다."}

    return {
        "status": "ok",
        "time": normalized_time,
        "result": f"알람 설정 완료: {normalized_time}",
    }


def send_message(contact: str, message: str) -> Dict[str, Any]:
    normalized_contact = (contact or "").strip()
    normalized_message = (message or "").strip()
    if not normalized_contact or not normalized_message:
        return {"error": "invalid_message", "message": "contact와 message가 필요합니다."}

    return {
        "status": "ok",
        "contact": normalized_contact,
        "message": normalized_message,
        "result": f"메시지 전송 완료: {normalized_contact}",
    }


def control_iot_device(device_id: str, action: str) -> Dict[str, Any]:
    """MQTT를 통해 스마트홈 기기를 제어하는 스캐폴딩 함수."""
    normalized_device = (device_id or "").strip()
    normalized_action = (action or "").strip().lower()
    if not normalized_device:
        return {"error": "invalid_device", "message": "device_id가 필요합니다."}
    if normalized_action not in {"on", "off", "dim"}:
        return {"error": "invalid_action", "message": "action은 on/off/dim 중 하나여야 합니다."}

    if mqtt_publish is None:
        return {
            "error": "mqtt_unavailable",
            "message": "MQTT 라이브러리가 설치되지 않았습니다.",
            "device_id": normalized_device,
            "action": normalized_action,
        }

    topic = f"smarthome/{normalized_device}/set"
    payload = normalized_action
    try:
        mqtt_publish.single(topic=topic, payload=payload, hostname="localhost", port=1883)
    except Exception as exc:  # pragma: no cover - runtime safety
        return {
            "error": "mqtt_publish_failed",
            "message": str(exc),
            "device_id": normalized_device,
            "action": normalized_action,
        }

    return {
        "status": "ok",
        "device_id": normalized_device,
        "action": normalized_action,
        "topic": topic,
    }


def update_avatar_state(emotion: str, gesture: str) -> Dict[str, Any]:
    """버추얼 캐릭터의 UI 상태를 갱신하는 스캐폴딩 함수."""
    _AVATAR_STATE["emotion"] = (emotion or "neutral").strip() or "neutral"
    _AVATAR_STATE["gesture"] = (gesture or "idle").strip() or "idle"
    return {
        "status": "ok",
        "avatar_state": dict(_AVATAR_STATE),
    }


class AudioManager:
    """STT, LLM 스트리밍, TTS 및 재생을 담당하는 오디오/미디어 관리자."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "dummy",
        model: str = "exaone3.5:2.4b",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = os.getenv("OLLAMA_MODEL", model or "exaone3.5:2.4b")
        self.vlm_model = os.getenv("OLLAMA_VLM_MODEL", "qwen2.5vl:3b")
        self.system_prompt = system_prompt or (
            "너는 한국어 음성비서야. "
            "사용자의 질문에 직접 답해. "
            "답변은 한국어 1~2문장만 말해. "
            "지시를 이해했다는 말, 예시 질문, 후속 질문 추천은 말하지 마. "
            "날씨, 알람, 검색, 기기 제어 같은 도구 요청은 코드가 먼저 처리하므로 일반 대화에서는 함수 호출 JSON을 만들지 마. "
            "미래 예언, 존재하지 않는 사실, 개인 비밀처럼 알 수 없는 질문만 절대 지어내지 말고, 반드시 '헤헤, 그건 지니가 아직 공부를 못해서 잘 몰라!'라고 자연스럽게 대답해라."
        )
        self._whisper_model = None
        self._whisper_lock = threading.Lock()
        self._tts_queue: Optional[queue.Queue] = None
        self._tts_thread: Optional[threading.Thread] = None
        self._tts_audio_buffer: Optional[io.BytesIO] = None
        self._tts_output_path: Optional[Path] = None
        self._tts_sentences: List[str] = []
        self._playback_lock = threading.Lock()
        self._active_playback_processes: List[subprocess.Popen] = []
        self.conversation_memory = ConversationMemoryBuffer(
            max_turns=int(os.getenv("CONVERSATION_MEMORY_TURNS", "6")),
            max_chars=int(os.getenv("CONVERSATION_MEMORY_CHARS", "1200")),
        )
        self.result_paths: List[str] = []
        self.request_timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
        self.vlm_timeout_seconds = float(os.getenv("OLLAMA_VLM_TIMEOUT_SECONDS", "180"))
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        self.max_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "512"))
        self.max_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "96"))
        self.max_tts_sentences = max(1, int(os.getenv("LLM_TTS_MAX_SENTENCES", "2")))
        self.tool_call_max_steps = int(os.getenv("OLLAMA_TOOL_MAX_STEPS", "4"))
        self.ollama_think = os.getenv("OLLAMA_THINK", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.tool_parser_fallback_enabled = True
        self.base_model_name = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
        self.lora_adapter_path = os.getenv("LORA_ADAPTER_PATH", "./saved_lora_model")
        self.use_lora_adapter = os.getenv("USE_LORA_ADAPTER", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.local_lora_model = None
        self.local_lora_tokenizer = None
        self._tool_circuit_breakers: Dict[str, CircuitBreaker] = {
            "get_visual_context": CircuitBreaker(),
            "get_weather": CircuitBreaker(),
            "control_iot_device": CircuitBreaker(),
            "get_current_weather": CircuitBreaker(),
            "search_web": CircuitBreaker(),
            "set_alarm": CircuitBreaker(),
            "send_message": CircuitBreaker(),
        }
        if self.use_lora_adapter:
            self._load_jetson_lora_model()

    def get_visual_context(self) -> Dict[str, Any]:
        """Fast loop 시각 맥락 조회 래퍼."""
        return get_visual_context()

    def control_iot_device(self, device_id: str, action: str) -> Dict[str, Any]:
        """IoT 제어 래퍼."""
        return control_iot_device(device_id, action)

    def update_avatar_state(self, emotion: str, gesture: str) -> Dict[str, Any]:
        """아바타 상태 갱신 래퍼."""
        return update_avatar_state(emotion, gesture)

    def preload_whisper_model(self, model_size: str = "tiny", compute_type: str = "float16") -> object:
        """Whisper 모델을 한 번만 로드합니다."""
        if self._whisper_model is not None:
            return self._whisper_model

        global WhisperModel
        if WhisperModel is None:
            try:
                import faster_whisper  # type: ignore

                WhisperModel = faster_whisper.WhisperModel
            except Exception:
                raise ImportError(
                    "faster-whisper가 설치되지 않았습니다. 먼저 'pip install faster-whisper'를 실행하세요."
                )

        with self._whisper_lock:
            if self._whisper_model is None:
                try:
                    self._whisper_model = WhisperModel(
                        model_size,
                        device="cuda",
                        compute_type=compute_type,
                    )
                except Exception:
                    self._whisper_model = WhisperModel(
                        model_size,
                        device="cpu",
                        compute_type="int8",
                    )
        return self._whisper_model

    def _load_jetson_lora_model(self):
        if not self.use_lora_adapter:
            return None
        if torch is None or PeftModel is None or AutoModelForCausalLM is None or AutoTokenizer is None or BitsAndBytesConfig is None:
            raise ImportError("PEFT/torch가 설치되지 않아 LoRA 추론을 사용할 수 없습니다.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        adapted_model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
        adapted_model.eval()
        self.local_lora_model = adapted_model
        self.local_lora_tokenizer = tokenizer
        return adapted_model

    def transcribe_audio_file(
        self,
        audio_path: str,
        model_size: str = "tiny",
        language: str = "ko",
        compute_type: str = "float16",
    ) -> str:
        """WAV 또는 MP3 파일을 한국어 텍스트로 변환합니다."""
        audio_path = str(Path(audio_path).expanduser().resolve())
        model = self.preload_whisper_model(model_size=model_size, compute_type=compute_type)
        segments, _ = model.transcribe(
            audio_path,
            language=language,
            beam_size=1,
            vad_filter=True,
        )
        transcript = "".join(segment.text for segment in segments).strip()
        self.run_memory_maintenance("post_stt")
        return transcript

    def transcribe(
        self,
        input_type: str = "mic",
        file_path: Optional[str] = None,
        language: str = "ko",
        timeout: float = 10.0,
        phrase_time_limit: float = 10.0,
    ) -> str:
        """입력 모드(mic/file)에 따라 음성을 텍스트로 변환합니다."""
        normalized_type = (input_type or "mic").strip().lower()

        if normalized_type == "mic":
            return self.transcribe_from_microphone(
                language=language,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )

        if normalized_type == "file":
            if not file_path:
                raise ValueError("file 모드에서는 file_path가 필요합니다.")
            return self.transcribe_from_audio_file(
                file_path=file_path,
                language=language,
            )

        raise ValueError(f"지원하지 않는 input_type입니다: {input_type}")

    def transcribe_from_audio_file(
        self,
        file_path: str,
        language: str = "ko",
    ) -> str:
        """speech_recognition.AudioFile로 파일 음성을 텍스트로 변환합니다."""
        if sr is None:
            return self.transcribe_audio_file(file_path, language=language)

        resolved_path = Path(file_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {resolved_path}")

        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(str(resolved_path)) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio, language=language)
        except (ValueError, OSError):
            # speech_recognition이 일부 포맷을 열지 못하면 기존 Whisper 경로로 폴백합니다.
            return self.transcribe_audio_file(str(resolved_path), language=language)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            raise RuntimeError(f"음성 인식 요청 실패: {exc}") from exc

    def transcribe_from_microphone(
        self,
        language: str = "ko",
        timeout: float = 10.0,
        phrase_time_limit: float = 10.0,
    ) -> str:
        """PC 마이크로 실시간 음성을 받아 텍스트로 변환합니다."""
        if sr is None:
            raise ImportError("speech_recognition이 설치되지 않았습니다. 먼저 'pip install SpeechRecognition pyaudio'를 실행하세요.")

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.2
        recognizer.non_speaking_duration = 0.5
        recognizer.phrase_threshold = 0.2

        try:
            microphone = sr.Microphone()
        except Exception as exc:
            raise RuntimeError(f"마이크 초기화에 실패했습니다: {exc}") from exc

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("[System] 마이크가 켜졌습니다. 말씀해 주세요!", flush=True)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        try:
            transcript = recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            print("[STT] 음성을 인식하지 못했습니다.", flush=True)
            return ""
        except sr.RequestError as exc:
            raise RuntimeError(f"음성 인식 요청 실패: {exc}") from exc

        return transcript

    def stream_remote_llm(self, prompt: str, image_inputs: Optional[List[Any]] = None):
        """Ollama 응답을 청크로 받아 문장 완성 시점마다 즉시 반환합니다."""
        if ollama is None:
            raise ImportError(
                "ollama 패키지가 설치되지 않았습니다. 먼저 'pip install ollama'를 실행하세요."
            )

        try:
            self.run_memory_maintenance("pre_llm")
            print("[LLM] Ollama 요청 시작", flush=True)
            print(f"[LLM] model={self.model}", flush=True)

            host = self.base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
            client = ollama.Client(host=host, timeout=self.request_timeout_seconds)
            print(f"[LLM] host={host}", flush=True)
            print(f"[LLM] timeout={self.request_timeout_seconds}s", flush=True)
            print(f"[LLM] keep_alive={self.ollama_keep_alive}", flush=True)
            ttft_start = time.perf_counter()

            raw_voice_text = self._extract_voice_text(prompt)
            voice_text = self._normalize_voice_transcript(raw_voice_text)
            llm_prompt = voice_text or self._normalize_prompt_for_llm(prompt)
            routing_prompt = voice_text or llm_prompt
            voice_instruction = self._build_voice_answer_instruction(routing_prompt)
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": f"{self.system_prompt}\n{voice_instruction}"},
                self._build_multimodal_user_message(llm_prompt, image_inputs),
            ]
            use_tools = self._should_use_tools(routing_prompt)
            refusal_text = self._should_refuse_prompt(routing_prompt)
            if refusal_text:
                print("[LLM] routing=refusal", flush=True)
                sanitized_refusal = self._sanitize_for_tts(refusal_text)
                if sanitized_refusal:
                    yield sanitized_refusal
                return
            builtin_answer = self._handle_builtin_tool_intent(routing_prompt) if use_tools and not self._model_supports_ollama_tools() else ""
            if builtin_answer:
                print("[LLM] routing=builtin-tool", flush=True)
                sanitized_answer = self._sanitize_for_tts(builtin_answer)
                if sanitized_answer:
                    yield sanitized_answer
                return
            if use_tools and not self._model_supports_ollama_tools():
                fallback_text = self._sanitize_for_tts("이 요청은 외부 도구가 필요하지만 현재 모델은 함수 호출을 지원하지 않습니다.")
                if fallback_text:
                    yield fallback_text
                return
            tools = self._build_ollama_tools() if use_tools else []
            print(f"[LLM] routing={'tool' if use_tools else 'direct'}", flush=True)

            if use_tools:
                try:
                    response = client.chat(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        stream=False,
                        keep_alive=self.ollama_keep_alive,
                        think=self.ollama_think,
                        options=self._build_ollama_options(),
                    )
                except Exception as exc:
                    logger.exception("[LLM] Ollama 호출 실패: %s", exc)
                    fallback_text = self._sanitize_for_tts("현재 네트워크 상태로 도구를 사용할 수 없습니다.")
                    if fallback_text:
                        yield fallback_text
                    return

                try:
                    response = self._resolve_tool_calls(client, messages, tools, response)
                except Exception as exc:
                    logger.exception("[LLM] Tool calling 단계 실패: %s", exc)
                    fallback_text = self._sanitize_for_tts("현재 네트워크 상태로 도구를 사용할 수 없습니다.")
                    if fallback_text:
                        yield fallback_text
                    return

                final_message = self._extract_ollama_message(response)
                final_tool_calls = self._extract_ollama_tool_calls(final_message)
                if final_tool_calls:
                    messages.append(final_message)

                preferred_korean_summary = self._extract_preferred_korean_summary(messages)
            else:
                response = None
                final_message = {"role": "assistant", "content": ""}
                preferred_korean_summary = ""

            try:
                chat_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": self.ollama_keep_alive,
                    "think": self.ollama_think,
                    "options": self._build_ollama_options(),
                }
                if use_tools:
                    chat_kwargs["tools"] = tools
                stream_response = client.chat(**chat_kwargs)
            except Exception as exc:
                logger.exception("[LLM] Ollama 스트리밍 호출 실패: %s", exc)
                fallback_text = self._sanitize_for_tts("현재 네트워크 상태로 도구를 사용할 수 없습니다.")
                if fallback_text:
                    yield fallback_text
                return

            assembled_text = ""
            sentence_buffer = ""
            chunk_count = 0
            text_chunk_count = 0
            yielded_count = 0
            ttft_logged = False
            last_emitted_text = ""

            for chunk in stream_response:
                chunk_count += 1
                chunk_message = self._extract_ollama_message(chunk)
                chunk_text = str(chunk_message.get("content", "") or "")
                if not chunk_text:
                    continue

                delta_text = self._extract_stream_delta(chunk_text, assembled_text)
                if not delta_text:
                    continue

                assembled_text += delta_text
                text_chunk_count += 1
                sentence_buffer += delta_text
                completed_sentences, sentence_buffer = self._split_complete_sentences(sentence_buffer)
                for sentence in completed_sentences:
                    cleaned_text = self._sanitize_for_tts(sentence)
                    if preferred_korean_summary and self._is_mostly_english(cleaned_text):
                        cleaned_text = self._sanitize_for_tts(preferred_korean_summary)
                    if not cleaned_text or cleaned_text == last_emitted_text:
                        continue

                    if not ttft_logged:
                        ttft_logged = True
                        ttft_latency = time.perf_counter() - ttft_start
                        print(f"[🚀 TTFT Latency: {ttft_latency:.2f}초]", flush=True)
                    yielded_count += 1
                    last_emitted_text = cleaned_text
                    print(f"[DEBUG] 전달할 LLM 텍스트: {cleaned_text}", flush=True)
                    yield cleaned_text
                    if yielded_count >= self.max_tts_sentences:
                        return

            if sentence_buffer.strip():
                cleaned_tail = self._sanitize_for_tts(sentence_buffer)
                if preferred_korean_summary and self._is_mostly_english(cleaned_tail):
                    cleaned_tail = self._sanitize_for_tts(preferred_korean_summary)
                if cleaned_tail and cleaned_tail != last_emitted_text:
                    if not ttft_logged:
                        ttft_logged = True
                        ttft_latency = time.perf_counter() - ttft_start
                        print(f"[🚀 TTFT Latency: {ttft_latency:.2f}초]", flush=True)
                    yielded_count += 1
                    last_emitted_text = cleaned_tail
                    print(f"[DEBUG] 전달할 LLM 텍스트: {cleaned_tail}", flush=True)
                    yield cleaned_tail
                    if yielded_count >= self.max_tts_sentences:
                        return

            if yielded_count == 0:
                fallback_text = self._sanitize_for_tts(str(final_message.get("content", "") or ""))
                if preferred_korean_summary and self._is_mostly_english(fallback_text):
                    fallback_text = self._sanitize_for_tts(preferred_korean_summary)
                if fallback_text and fallback_text != last_emitted_text:
                    if not ttft_logged:
                        ttft_logged = True
                        ttft_latency = time.perf_counter() - ttft_start
                        print(f"[🚀 TTFT Latency: {ttft_latency:.2f}초]", flush=True)
                    yielded_count += 1
                    last_emitted_text = fallback_text
                    print(f"[DEBUG] 전달할 LLM 텍스트(폴백): {fallback_text}", flush=True)
                    yield fallback_text

            if yielded_count == 0:
                print(
                    f"[LLM] 경고: 스트리밍 응답에서 문장을 추출하지 못했습니다. chunk_count={chunk_count}, text_chunk_count={text_chunk_count}",
                    flush=True,
                )
        except GeneratorExit:
            raise
        except Exception as exc:
            logger.exception("[LLM] 스트리밍 처리 외곽 실패: %s", exc)
            fallback_text = self._sanitize_for_tts("현재 네트워크 상태로 도구를 사용할 수 없습니다.")
            if fallback_text:
                yield fallback_text
        finally:
            self.run_memory_maintenance("post_llm")

    def stream_remote_vlm(self, prompt: str, image_inputs: Optional[List[Any]] = None):
        """이미지 입력은 VLM 전용 Ollama 모델로 처리합니다."""
        if not image_inputs:
            return self.stream_remote_llm(prompt=prompt, image_inputs=image_inputs)

        def _generator():
            original_model = self.model
            original_timeout = self.request_timeout_seconds
            self.model = self.vlm_model
            self.request_timeout_seconds = self.vlm_timeout_seconds
            try:
                yield from self.stream_remote_llm(prompt=prompt, image_inputs=image_inputs)
            finally:
                self.model = original_model
                self.request_timeout_seconds = original_timeout

        return _generator()

    def _build_ollama_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "현재 날씨 조회. location이 없으면 Daejeon 사용.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "도시명. 예: Daejeon",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
            ,
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "최신 정보 검색. query에 검색어를 넣는다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색어",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": "알람 또는 타이머 설정. time에 설정할 시간을 넣는다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time": {
                                "type": "string",
                                "description": "알람 시간 또는 지속시간",
                            }
                        },
                        "required": ["time"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "문자 전송을 흉내낸다. contact와 message가 필요하다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contact": {
                                "type": "string",
                                "description": "연락처나 이름",
                            },
                            "message": {
                                "type": "string",
                                "description": "보낼 메시지",
                            },
                        },
                        "required": ["contact", "message"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _build_multimodal_user_message(self, prompt: str, image_inputs: Optional[List[Any]]) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        images = self._normalize_ollama_images(image_inputs)
        if images:
            message["images"] = images
        return message

    def _normalize_ollama_images(self, image_inputs: Optional[List[Any]]) -> List[str]:
        if not image_inputs:
            return []

        normalized: List[str] = []
        for image_input in image_inputs:
            if isinstance(image_input, bytes):
                normalized.append(base64.b64encode(image_input).decode("utf-8"))
                continue

            if isinstance(image_input, str):
                candidate = image_input.strip()
                if not candidate:
                    continue

                possible_path = Path(candidate).expanduser()
                if possible_path.exists() and possible_path.is_file():
                    file_bytes = possible_path.read_bytes()
                    normalized.append(base64.b64encode(file_bytes).decode("utf-8"))
                    continue

                normalized.append(candidate)

        return normalized

    def _model_supports_ollama_tools(self) -> bool:
        configured = os.getenv("OLLAMA_SUPPORTS_TOOLS")
        if configured is not None:
            return configured.strip().lower() in {"1", "true", "yes", "on"}
        return "exaone" not in self.model.lower()

    def _handle_builtin_tool_intent(self, voice_text: str) -> str:
        normalized = self._normalize_voice_transcript(voice_text)
        if not normalized:
            return ""

        if re.search(r"(날씨|기온|비 와|눈 와|우산)", normalized):
            query_location, spoken_location = self._extract_weather_location(normalized)
            weather = get_weather(query_location)
            if weather.get("error"):
                return "날씨 정보를 불러오는 데 실패했습니다. 통신 상태를 확인해주세요."
            weather_ko = str(weather.get("weather_ko") or "정보 없음")
            temperature_c = weather.get("temperature_c")
            if temperature_c is None:
                return f"현재 {spoken_location}의 날씨는 {weather_ko}입니다."
            return f"현재 {spoken_location}의 날씨는 {weather_ko}이고 기온은 {temperature_c}도씨입니다."

        if re.search(r"(알람|타이머)", normalized):
            match = re.search(r"((?:내일|오늘)?\s*(?:오전|오후)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?|\d+\s*(?:분|시간)\s*(?:뒤|후)?)", normalized)
            time_value = match.group(1).strip() if match else "요청한 시간"
            result = set_alarm(time_value)
            return str(result.get("result") or f"알람 설정 완료: {time_value}")

        device_command = self._extract_device_command(normalized)
        if device_command:
            device_id, action, spoken_device = device_command
            result = control_iot_device(device_id, action)
            if result.get("error"):
                return f"{spoken_device} 제어 요청은 이해했지만, 현재 장치 제어 모듈이 연결되어 있지 않습니다."
            action_ko = "켰습니다" if action == "on" else "껐습니다" if action == "off" else "조절했습니다"
            return f"{spoken_device}를 {action_ko}."

        return ""

    def _extract_device_command(self, voice_text: str) -> tuple[str, str, str] | None:
        normalized = re.sub(r"\s+", " ", voice_text or "").strip()
        if not normalized:
            return None

        device_aliases = (
            ("블루투스 스피커", "bluetooth_speaker"),
            ("스피커", "speaker"),
            ("블루투스", "bluetooth"),
            ("조명", "light"),
            ("전등", "light"),
            ("플러그", "plug"),
        )
        action = "on" if re.search(r"(켜|켜줘|켜라|on)", normalized, re.IGNORECASE) else "off" if re.search(r"(꺼|꺼줘|꺼라|off)", normalized, re.IGNORECASE) else ""
        if not action:
            return None

        for spoken_device, device_id in device_aliases:
            if spoken_device in normalized:
                return device_id, action, spoken_device
        return None

    def _extract_weather_location(self, voice_text: str) -> tuple[str, str]:
        normalized = re.sub(r"\s+", " ", voice_text or "").strip()
        location_aliases = {
            "서울": ("Seoul", "서울"),
            "대전": ("Daejeon", "대전"),
            "부산": ("Busan", "부산"),
            "대구": ("Daegu", "대구"),
            "인천": ("Incheon", "인천"),
            "광주": ("Gwangju", "광주"),
            "울산": ("Ulsan", "울산"),
            "제주": ("Jeju", "제주"),
        }
        for keyword, location in location_aliases.items():
            if keyword in normalized:
                return location
        return "Daejeon", "대전"

    def _extract_voice_text(self, prompt: str) -> str:
        text = str(prompt or "").strip()
        if not text:
            return ""

        match = re.search(r"\[voice\]\s*(.*?)(?:\n\s*\[visual_context\]|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
        return text

    def _normalize_voice_transcript(self, transcript: str) -> str:
        normalized = re.sub(r"\s+", " ", transcript or "").strip()
        if not normalized:
            return ""

        replacements = (
            ("물이 생활", "무리 생활"),
            ("물의 생활", "무리 생활"),
            ("무리생활", "무리 생활"),
        )
        for source, target in replacements:
            normalized = normalized.replace(source, target)

        return normalized

    def _classify_voice_question(self, voice_text: str) -> str:
        normalized = re.sub(r"\s+", " ", voice_text or "").strip()
        if not normalized:
            return "발화가 비어 있으면 다시 말해 달라고 짧게 요청한다."
        if re.search(r"(알람|타이머|문자|메시지|켜|꺼|조명|전등|플러그)", normalized):
            return "기기 제어 또는 작업 실행 요청이다. 지원 도구가 있으면 도구를 사용하고, 없으면 할 수 있는 범위를 짧게 말한다."
        if re.search(r"(날씨|기온|비 와|눈 와|우산)", normalized):
            return "실시간 날씨 요청이다. 날씨 도구를 사용하고, 실패하면 임의로 날씨를 만들지 않는다."
        if re.search(r"(최신|뉴스|주가|환율|검색|오늘 경기|현재)", normalized):
            return "최신 정보 요청이다. 외부 정보가 필요하면 검색 도구를 사용하고, 도구 결과가 없으면 추측하지 않는다."
        if re.search(r"(점심|저녁|아침|메뉴|먹을까|먹지|추천)", normalized):
            return "일상 추천 요청이다. 사용자의 취향 정보가 부족하면 무난한 선택지 2~3개를 제안하고, 하나를 가볍게 추천한다."
        if re.search(r"(회사|기업|그룹|브랜드|자동차|전자|은행|대학교|기관)", normalized):
            return "기업, 기관, 브랜드에 대한 일반 지식 질문이다. 확실한 핵심 정의만 말하고, 불확실한 계열사나 세부 정보는 덧붙이지 않는다."
        if re.search(r"(뭐야|무엇|누구|어디|설명|알려줘)", normalized):
            return "개념 설명 또는 일반 지식 질문이다. 오래 변하지 않는 기본 사실을 중심으로 1~2문장으로 설명한다."
        if re.search(r"(왜|이유|어떻게|원리)", normalized):
            return "이유 설명 질문이다. '왜 X는 Y를 해?' 형태라면 Y를 한다는 질문의 전제를 함부로 부정하지 말고, 그 이유와 장점을 설명한다."
        return "일반 대화 또는 일반 지식 질문이다. 사용자의 의도를 가장 자연스럽게 해석해 짧게 답한다."

    def _build_voice_answer_instruction(self, voice_text: str) -> str:
        question_profile = self._classify_voice_question(voice_text)
        parts = [
            "음성 인식 결과에는 조사, 띄어쓰기, 비슷한 발음의 단어 오류가 있을 수 있다. 문맥상 자연스럽게 해석해.",
            f"질문 유형: {question_profile}",
            "일반 지식, 개념 설명, 음식/활동 추천, 가벼운 조언은 바로 답해. "
            "기업 설명은 확실한 핵심 정의만 말해. 추천 질문은 선택지 2~3개와 무난한 하나를 제안해. "
            "사실관계를 반대로 말하지 말고, 사용자가 말한 숫자와 단위를 바꾸지 마. "
            "한국어 1~2문장으로만 답해.",
        ]
        return "\n".join(parts)

    def _normalize_prompt_for_llm(self, prompt: str) -> str:
        text = str(prompt or "").strip()
        raw_voice_text = self._extract_voice_text(text)
        voice_text = self._normalize_voice_transcript(raw_voice_text)
        answer_instruction = self._build_voice_answer_instruction(voice_text)
        visual_match = re.search(r"\[visual_context\]\s*(.*)\Z", text, flags=re.DOTALL | re.IGNORECASE)
        if not visual_match:
            return (
                f"사용자 발화: {voice_text}\n"
                f"{answer_instruction}"
            )

        visual_text = re.sub(r"\s+", " ", visual_match.group(1)).strip()
        if not visual_text:
            return (
                f"사용자 발화: {voice_text}\n"
                f"{answer_instruction}"
            )

        return (
            f"사용자 발화: {voice_text}\n"
            f"{answer_instruction}\n"
            f"참고 시각 정보: {visual_text}\n"
            "사용자 발화에만 자연스럽게 답하고, 참고 시각 정보나 내부 형식은 말하지 마."
        )

    def _resolve_tool_calls(self, client: Any, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], initial_response: Any) -> Any:
        response = initial_response
        max_steps = self.tool_call_max_steps

        for step in range(max_steps):
            message = self._extract_ollama_message(response)
            tool_calls = self._extract_ollama_tool_calls(message)
            if not tool_calls:
                print(f"[LLM] tool_call_step={step}: 도구 호출 없음, 최종 응답 사용", flush=True)
                return response

            print(f"[LLM] tool_call_step={step}: tool_calls={len(tool_calls)}", flush=True)
            messages.append(message)
            for tool_call in tool_calls:
                function_info = tool_call.get("function", {})
                function_name = str(function_info.get("name", ""))
                raw_arguments = function_info.get("arguments", {})
                print(f"[LLM] tool_call_step={step}: function={function_name}", flush=True)

                if isinstance(raw_arguments, str):
                    try:
                        function_args = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        extracted = self._extract_json_from_text(raw_arguments)
                        function_args = extracted if isinstance(extracted, dict) else {}
                elif isinstance(raw_arguments, dict):
                    function_args = raw_arguments
                else:
                    function_args = {}

                tool_result = self._dispatch_tool_call(function_name, function_args)
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            response = client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=False,
                keep_alive=self.ollama_keep_alive,
                think=self.ollama_think,
                options=self._build_ollama_options(),
            )

        raise RuntimeError("Ollama tool calling step limit exceeded")

    def _build_ollama_options(self) -> Dict[str, Any]:
        return {
            "temperature": 0.0,
            "num_predict": self.max_num_predict,
            "top_p": 0.9,
            "num_ctx": self.max_num_ctx,
        }

    def run_memory_maintenance(self, stage: str = "") -> None:
        """엣지 디바이스 메모리 파편화를 줄이기 위한 정리 루틴."""
        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if stage:
            logger.debug("memory_maintenance_completed stage=%s", stage)

    def _should_use_tools(self, prompt: str) -> bool:
        normalized_prompt = (prompt or "").lower()
        tool_keywords = (
            "최신",
            "근황",
            "검색",
            "위키",
            "날씨",
            "기온",
            "우산",
            "비",
            "온도",
            "섭씨",
            "화씨",
            "알람",
            "타이머",
            "메시지",
            "문자",
            "보내",
            "연락",
            "weather",
            "device",
            "기기",
            "조명",
            "전등",
            "플러그",
            "스마트",
            "iot",
            "켜",
            "꺼",
        )
        return any(keyword in normalized_prompt for keyword in tool_keywords)

    def _should_refuse_prompt(self, prompt: str) -> str:
        normalized_prompt = (prompt or "").lower()
        refusal_patterns = (
            "100만 광년",
            "외계인",
            "미래",
            "복권",
            "로또",
            "예언",
            "없는 이름",
            "존재하지 않는",
        )
        if any(pattern in normalized_prompt for pattern in refusal_patterns):
            return "헤헤, 그건 지니가 아직 공부를 못해서 잘 몰라!"
        return ""

    def _extract_ollama_message(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "message"):
            message = response.message
            if hasattr(message, "model_dump"):
                return dict(message.model_dump())
            if isinstance(message, dict):
                return dict(message)
            return {
                "role": getattr(message, "role", "assistant") or "assistant",
                "content": getattr(message, "content", "") or "",
                "tool_calls": self._normalize_tool_calls(getattr(message, "tool_calls", None)),
            }

        if isinstance(response, dict):
            message = response.get("message") or {}
            if isinstance(message, dict):
                return dict(message)
            if isinstance(message, str):
                extracted = self._extract_json_from_text(message)
                if isinstance(extracted, dict):
                    return extracted

        return {"role": "assistant", "content": ""}

    def _extract_ollama_tool_calls(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = message.get("tool_calls") or []
        normalized = self._normalize_tool_calls(calls)
        if normalized:
            return normalized

        content = str(message.get("content", "") or "")
        extracted_calls = self._extract_tool_calls_from_text(content)
        if extracted_calls:
            return extracted_calls

        return []

    def _normalize_tool_calls(self, calls: Any) -> List[Dict[str, Any]]:
        if not isinstance(calls, list):
            if isinstance(calls, str):
                parsed = self._extract_tool_calls_from_text(calls)
                return parsed
            if isinstance(calls, dict):
                return [calls]
            return []

        normalized: List[Dict[str, Any]] = []
        for call in calls:
            if isinstance(call, dict):
                normalized.append(call)
                continue

            if hasattr(call, "model_dump"):
                dumped = call.model_dump()
                if isinstance(dumped, dict):
                    normalized.append(dumped)
                    continue

            function_obj = getattr(call, "function", None)
            if function_obj is not None:
                function_payload = {
                    "name": getattr(function_obj, "name", "") or "",
                    "arguments": getattr(function_obj, "arguments", {}) or {},
                }
                normalized.append({"function": function_payload})

        return normalized

    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        candidate_text = (text or "").strip()
        if not candidate_text:
            return None

        patterns = [
            r"\{.*?\}",
            r"\[.*?\]",
        ]
        for pattern in patterns:
            match = re.search(pattern, candidate_text, flags=re.DOTALL)
            if not match:
                continue
            fragment = match.group(0)
            try:
                return json.loads(fragment)
            except JSONDecodeError:
                continue

        return None

    def _extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        extracted = self._extract_json_from_text(text)
        if isinstance(extracted, list):
            return self._normalize_tool_calls(extracted)
        if isinstance(extracted, dict):
            if "tool_calls" in extracted:
                return self._normalize_tool_calls(extracted.get("tool_calls"))
            if "function" in extracted:
                return self._normalize_tool_calls([extracted])
            return []
        return []

    def _extract_preferred_korean_summary(self, messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            content = str(message.get("content", "") or "")
            extracted: Optional[Any]
            try:
                extracted = json.loads(content)
            except JSONDecodeError:
                extracted = self._extract_json_from_text(content)
            if isinstance(extracted, dict):
                summary_ko = str(extracted.get("summary_ko", "") or "").strip()
                if summary_ko:
                    return summary_ko
        return ""

    def _is_mostly_english(self, text: str) -> bool:
        if not text:
            return False

        hangul_count = len(re.findall(r"[가-힣]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        return latin_count > 0 and hangul_count < max(3, latin_count // 4)

    def _extract_ollama_text(self, response: Any) -> str:
        message = self._extract_ollama_message(response)
        content = message.get("content", "")
        return str(content) if content else ""

    def _split_complete_sentences(self, text: str) -> tuple[List[str], str]:
        completed: List[str] = []
        normalized = text.replace("\r\n", "\n")
        cursor = 0

        for idx, ch in enumerate(normalized):
            is_terminal = ch in ".!?。！？\n"
            if not is_terminal:
                continue

            segment = normalized[cursor:idx + 1]
            if ch == "\n":
                segment = normalized[cursor:idx]

            sentence = segment.strip()
            if sentence:
                completed.append(sentence)
            cursor = idx + 1

        return completed, normalized[cursor:]

    def _sanitize_for_tts(self, text: str) -> str:
        cleaned = self._clean_llm_text(text)
        cleaned = re.sub("[\U00010000-\U0010FFFF]", " ", cleaned)
        cleaned = re.sub("[\u2600-\u26FF\u2700-\u27BF]", " ", cleaned)
        cleaned = re.sub(r"[★☆◆◇■□●○▶▷◀◁※]+", " ", cleaned)
        cleaned = re.sub(r"\b\d+(?:\.\d+)?\s?°?\s?[Ff]\b", " ", cleaned)
        cleaned = re.sub(r"\b(?:fahrenheit|화씨)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?i)\bmarkdown\b", " ", cleaned)
        cleaned = re.sub(r"\b마침표\b", " ", cleaned)
        cleaned = re.sub(r"\b느낌표\b", " ", cleaned)
        cleaned = re.sub(r"[#+*_`~]+", " ", cleaned)
        cleaned = re.sub(r"[\u4e00-\u9fff]+", " ", cleaned)
        cleaned = re.sub(r"\s*[,，]+\s*([.!?。！？])", r"\1", cleaned)
        cleaned = re.sub(r"([.!?。！？]){2,}", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _extract_stream_delta(self, chunk_text: str, assembled_text: str) -> str:
        """스트리밍 청크가 누적형/델타형 어느 포맷이든 신규 텍스트만 반환합니다."""
        if not chunk_text:
            return ""

        # 누적형 스트림(매 청크에 전체 텍스트 포함) 처리
        if assembled_text and chunk_text.startswith(assembled_text):
            return chunk_text[len(assembled_text):]

        # 완전히 동일한 청크 재전송은 무시
        if assembled_text and chunk_text == assembled_text:
            return ""

        # 부분 중첩(접미-접두) 케이스를 제거해 중복 발화를 예방
        max_overlap = min(len(assembled_text), len(chunk_text))
        for overlap in range(max_overlap, 0, -1):
            if assembled_text.endswith(chunk_text[:overlap]):
                return chunk_text[overlap:]

        # 기본적으로 델타형 스트림은 청크 전체가 신규 텍스트입니다.
        return chunk_text

    def _dispatch_tool_call(self, function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
        if function_name == "get_current_weather":
            return self._call_tool_with_resilience(
                function_name,
                get_current_weather,
                location=str(function_args.get("location", "Daejeon")),
            )
        if function_name == "get_weather":
            return self._call_tool_with_resilience(
                function_name,
                get_weather,
                location=str(function_args.get("location", "Daejeon")),
            )
        if function_name == "get_visual_context":
            return self._call_tool_with_resilience(
                function_name,
                get_visual_context,
            )
        if function_name == "control_iot_device":
            return self._call_tool_with_resilience(
                function_name,
                control_iot_device,
                device_id=str(function_args.get("device_id", "")),
                action=str(function_args.get("action", "")),
            )
        if function_name == "update_avatar_state":
            return update_avatar_state(
                str(function_args.get("emotion", "neutral")),
                str(function_args.get("gesture", "idle")),
            )
        if function_name == "search_web":
            return self._call_tool_with_resilience(
                function_name,
                search_web,
                query=str(function_args.get("query", "")),
            )
        if function_name == "set_alarm":
            return self._call_tool_with_resilience(
                function_name,
                set_alarm,
                time_value=str(function_args.get("time", "")),
            )
        if function_name == "send_message":
            return self._call_tool_with_resilience(
                function_name,
                send_message,
                contact=str(function_args.get("contact", "")),
                message=str(function_args.get("message", "")),
            )

        return {
            "error": "unknown_tool",
            "message": f"지원하지 않는 함수 호출: {function_name}",
        }

    def _call_tool_with_resilience(self, tool_name: str, tool_fn: Any, **kwargs: Any) -> Dict[str, Any]:
        breaker = self._tool_circuit_breakers.get(tool_name)
        if breaker is not None and not breaker.allow_request():
            return self._offline_fallback(tool_name, "circuit_open")

        delay = 0.2
        max_attempts = 1 if tool_name in {"get_weather", "get_current_weather"} else 2
        for attempt in range(max_attempts):
            try:
                result = tool_fn(**kwargs)
                if isinstance(result, dict) and result.get("error"):
                    if result.get("error") == "timeout":
                        if breaker is not None:
                            breaker.record_failure()
                        return self._offline_fallback(tool_name, "timeout")
                    raise RuntimeError(str(result.get("error")))
                if breaker is not None:
                    breaker.record_success()
                if isinstance(result, dict):
                    return result
                return {"status": "ok", "result": result}
            except Exception as exc:  # pragma: no cover - runtime safety
                if breaker is not None:
                    breaker.record_failure()
                if attempt == max_attempts - 1:
                    return self._offline_fallback(tool_name, str(exc))
                time.sleep(delay)
                delay *= 2

        return self._offline_fallback(tool_name, "unknown")

    def _offline_fallback(self, tool_name: str, reason: str) -> Dict[str, Any]:
        if tool_name == "get_visual_context":
            return {
                "error": "offline_fallback",
                "reason": reason,
                "presence": False,
                "face_expression": "unknown",
                "gaze_direction": "unknown",
                "upper_body_posture": "unknown",
                "source": "offline_mode",
                "sensor_status": "disconnected",
            }
        if tool_name == "control_iot_device":
            return {
                "error": "offline_fallback",
                "reason": reason,
                "status": "queued_offline",
                "hardware_status": "controller_unreachable",
            }
        if tool_name in {"get_weather", "get_current_weather"}:
            return {
                "error": "offline_fallback",
                "reason": reason,
                "location": "unknown",
                "temperature_c": None,
                "weather": "정보 없음",
            }
        return {
            "error": "offline_fallback",
            "reason": reason,
            "tool": tool_name,
        }

    def _clean_llm_text(self, text: str) -> str:
        """생각 과정, 메타 문구, 구조적 표기를 제거하고 답변 본문만 남깁니다."""
        if not text:
            return ""

        cleaned = text.strip()
        cleaned = cleaned.replace("\n", " ").replace("\r", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"헤[a-zA-Z]+", "헤헤", cleaned)
        if "지니가 아직" not in cleaned:
            cleaned = re.sub(r"^헤헤,?\s*", "", cleaned)
        cleaned = re.sub(r"```.*?```", " ", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"[\[\](){}<>]", "", cleaned)
        cleaned = re.sub(r"(?<!\w)[*_>#-]+", " ", cleaned)
        cleaned = re.sub(r'(?i)\b(?:thinking|reasoning|analysis|meta|note|answer|response|final|summary|assistant)\s*[:：]', " ", cleaned)
        cleaned = re.sub(r"(?i)\b(?:answer|response|final|summary)\b", "", cleaned)
        cleaned = re.sub(r"(?:답변|응답|최종|요약)\s*[:：]", " ", cleaned)
        cleaned = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", cleaned)
        cleaned = cleaned.replace("\"", "").replace("'", "")

        candidates = re.split(r"(?<=[.?!。！？])\s+", cleaned)
        kept_sentences = []
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            candidate = re.sub(r"\s+", " ", candidate)
            lowered = candidate.lower()
            if not re.search(r"[가-힣a-zA-Z]", candidate):
                continue
            if re.search(r"[\u4e00-\u9fff]", candidate):
                continue

            if any(
                marker in lowered
                for marker in [
                    "thinking",
                    "reasoning",
                    "analysis",
                    "meta",
                    "note",
                    "here is",
                    "here’s",
                    "answer:",
                    "response:",
                    "bullet",
                    "list",
                    "code",
                    "생각",
                    "과정",
                    "분석",
                    "메타",
                    "user input",
                    "user says",
                    "analyze",
                    "analyze user",
                    "추천 질문",
                    "질문 예시",
                ]
            ):
                continue

            if candidate.startswith(("- ", "* ")):
                continue

            if re.fullmatch(r"[A-Za-z0-9\s:.\-]+", candidate) and len(candidate.split()) <= 4:
                continue

            kept_sentences.append(candidate)

        if not kept_sentences:
            return ""

        cleaned = " ".join(kept_sentences[:2])
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned and cleaned[-1] not in ".!?。！？":
            cleaned = cleaned + "."

        return cleaned.strip()

    def synthesize_speech(self, text: str, output_path: str = "response.mp3") -> str:
        """문장을 한국어 오디오 파일로 저장합니다."""
        safe_text = self._sanitize_for_tts(text)
        print(f"[TTS] 합성할 문장: {safe_text}", flush=True)
        tts = gTTS(text=safe_text, lang="ko", slow=False)
        tts.save(output_path)
        return output_path

    def synthesize_speech_bytes(self, text: str) -> bytes:
        """문장을 한국어 MP3 바이트로 메모리에서 합성합니다."""
        safe_text = self._sanitize_for_tts(text)
        print(f"[TTS] 합성할 문장: {safe_text}", flush=True)
        memory_buffer = io.BytesIO()
        try:
            tts = gTTS(text=safe_text, lang="ko", slow=False)
            tts.write_to_fp(memory_buffer)
            audio_bytes = memory_buffer.getvalue()
            self._release_tts_tensor_if_needed(audio_bytes)
            return audio_bytes
        finally:
            self.run_memory_maintenance("post_tts_sentence")

    def _release_tts_tensor_if_needed(self, audio_object: Any) -> None:
        """딥러닝 TTS로 교체될 경우를 대비해 텐서 리소스를 즉시 CPU로 내립니다."""
        if torch is None:
            return
        if isinstance(audio_object, torch.Tensor):
            _ = audio_object.detach().cpu()

    def _play_audio_file(self, audio_path: str) -> None:
        """Ubuntu/Linux에서 오디오를 백그라운드로 재생합니다."""
        process: Optional[subprocess.Popen] = None
        try:
            process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                process = subprocess.Popen(
                    ["aplay", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                print(f"오디오 재생기(ffplay/aplay)를 찾을 수 없습니다: {audio_path}")
        if process is not None:
            with self._playback_lock:
                self._active_playback_processes.append(process)

    def halt_tts_playback(self) -> None:
        """Barge-in 시 대기 중인 TTS와 현재 재생 중인 오디오를 중단합니다."""
        if self._tts_queue is not None:
            while True:
                try:
                    self._tts_queue.get_nowait()
                    self._tts_queue.task_done()
                except queue.Empty:
                    break

        with self._playback_lock:
            processes = list(self._active_playback_processes)
            self._active_playback_processes.clear()
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=0.5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        self.run_memory_maintenance("barge_in_halt")

    def _tts_worker(self) -> None:
        """문장 단위 응답을 모아 최종 TTS 입력으로 보존합니다."""
        assert self._tts_queue is not None
        while True:
            sentence = self._tts_queue.get()
            if sentence is None:
                self._tts_queue.task_done()
                break

            safe_sentence = self._sanitize_for_tts(str(sentence))
            if safe_sentence:
                self._tts_sentences.append(safe_sentence)
            self._tts_queue.task_done()

    def start_tts_worker(self, output_dir: Path, output_prefix: str) -> None:
        """문장 재생용 TTS 워커를 시작합니다."""
        self.result_paths = []
        self._tts_output_path = output_dir / f"{output_prefix}.mp3"
        self._tts_audio_buffer = None
        self._tts_sentences = []
        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True,
        )
        self._tts_thread.start()

    def enqueue_sentence(self, sentence: str) -> None:
        """실시간 재생을 위해 문장을 큐에 넣습니다."""
        if self._tts_queue is None:
            raise RuntimeError("TTS 워커가 시작되지 않았습니다.")
        self._tts_queue.put(sentence)

    def stop_tts_worker(self) -> None:
        """TTS 워커를 종료하고 남은 작업을 정리합니다."""
        if self._tts_queue is not None:
            self._tts_queue.put(None)
            self._tts_queue.join()
        if self._tts_thread is not None:
            self._tts_thread.join(timeout=2)

        if self._tts_output_path is not None and self._tts_sentences:
            full_text = self._sanitize_for_tts(" ".join(self._tts_sentences))
            if full_text:
                self._tts_output_path.parent.mkdir(parents=True, exist_ok=True)
                self.synthesize_speech(full_text, str(self._tts_output_path))
                self.result_paths = [str(self._tts_output_path)]
                print(f"[TTS 출력] : 음성 합성 완료: {self._tts_output_path}", flush=True)
                self._play_audio_file(str(self._tts_output_path))

        self._tts_audio_buffer = None
        self._tts_output_path = None
        self._tts_sentences = []
        self._tts_queue = None
        self._tts_thread = None
        self.run_memory_maintenance("post_tts")

    def close(self) -> None:
        """오디오/재생 관련 백그라운드 리소스를 정리합니다."""
        self.halt_tts_playback()
        self.stop_tts_worker()
