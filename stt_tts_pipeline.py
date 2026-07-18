from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import requests

from agents import RAGAgent, TOOL_CALL_SYSTEM_PROMPT, ToolAgent, ToolCall, extract_tool_call_from_text
from audio_manager import AudioManager
from main import run_pipeline


DEFAULT_MIC_TIMEOUT = 10.0
DEFAULT_MIC_PHRASE_TIME_LIMIT = 10.0
DEFAULT_QUEUE_MAXSIZE = 8
DEFAULT_FIRST_CHUNK_MIN_CHARS = 10
DEFAULT_CHUNK_MIN_CHARS = 24
DEFAULT_CHUNK_MAX_CHARS = 72
DEFAULT_PLAYBACK_PREBUFFER_CHUNKS = 1
DEFAULT_RAG_CONTEXT_CHARS = 600
DEFAULT_MEMORY_TURNS = 3


DEFAULT_OLLAMA_MODEL = "exaone3.5:2.4b"


def _split_tts_units(
    text: str,
    min_chars: int,
    max_chars: int,
    force: bool = False,
) -> tuple[list[str], str]:
    buffer = re.sub(r"\s+", " ", text or " ").strip()
    if not buffer:
        return [], ""

    units: list[str] = []
    cursor = 0
    for index, char in enumerate(buffer):
        length = index - cursor + 1
        is_terminal = char in ".!?。！？"
        is_soft_break = char in ",，;；" and length >= min_chars
        if not is_terminal and not is_soft_break:
            continue
        unit = buffer[cursor:index + 1].strip()
        if unit:
            units.append(unit)
        cursor = index + 1

    remainder = buffer[cursor:].strip()
    if len(remainder) >= max_chars:
        split_at = max(remainder.rfind(" ", 0, max_chars), remainder.rfind(",", 0, max_chars), remainder.rfind("，", 0, max_chars))
        if split_at <= 0:
            split_at = max_chars
        units.append(remainder[:split_at].strip())
        remainder = remainder[split_at:].strip()

    if force and remainder:
        units.append(remainder)
        remainder = ""

    return [unit for unit in units if unit], remainder


def _play_audio_file_blocking(audio_path: str, stop_event: Optional[asyncio.Event] = None) -> None:
    players = (
        ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", audio_path],
        ["aplay", audio_path],
    )
    for command in players:
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while process.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return
                time.sleep(0.03)
            return
        except FileNotFoundError:
            continue
    print(f"오디오 재생기(ffplay/aplay)를 찾을 수 없습니다: {audio_path}", flush=True)


def _drain_async_queue(target_queue: asyncio.Queue[Any]) -> None:
    while True:
        try:
            target_queue.get_nowait()
            target_queue.task_done()
        except asyncio.QueueEmpty:
            break


class StreamingVoicePipeline:
    """Low-latency STT -> LLM -> TTS runtime with async queue backpressure."""

    def __init__(
        self,
        output_path: str = "response.mp3",
        base_url: Optional[str] = None,
        api_key: str = "dummy",
        model: str = DEFAULT_OLLAMA_MODEL,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        first_chunk_min_chars: int = DEFAULT_FIRST_CHUNK_MIN_CHARS,
        chunk_min_chars: int = DEFAULT_CHUNK_MIN_CHARS,
        chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
        playback_prebuffer_chunks: int = DEFAULT_PLAYBACK_PREBUFFER_CHUNKS,
        rag_corpus_path: str = "./knowledge_base",
        rag_context_chars: int = DEFAULT_RAG_CONTEXT_CHARS,
        enable_rag: bool = True,
    ) -> None:
        self.audio_manager = AudioManager(base_url=base_url, api_key=api_key, model=model)
        self.output_path = Path(output_path)
        self.queue_maxsize = queue_maxsize
        self.first_chunk_min_chars = first_chunk_min_chars
        self.chunk_min_chars = chunk_min_chars
        self.chunk_max_chars = chunk_max_chars
        self.playback_prebuffer_chunks = playback_prebuffer_chunks
        self.enable_rag = enable_rag
        self.rag_context_chars = rag_context_chars
        self.tool_agent = ToolAgent()
        self.rag_agent = RAGAgent(corpus_path=rag_corpus_path, max_context_chars=rag_context_chars)
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="voice-pipeline")

    async def transcribe_once(
        self,
        input_type: str = "mic",
        file_path: Optional[str] = None,
        timeout: float = DEFAULT_MIC_TIMEOUT,
        phrase_time_limit: float = DEFAULT_MIC_PHRASE_TIME_LIMIT,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.audio_manager.transcribe(
                input_type=input_type,
                file_path=file_path,
                language="ko",
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            ),
        )

    async def stream_llm_chunks(self, prompt: str) -> AsyncIterator[str]:
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=self.queue_maxsize)
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                for chunk in self.audio_manager.stream_remote_llm(prompt):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        producer_task = loop.run_in_executor(self._executor, _producer)
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            await producer_task

    async def stream_llm_raw(self, prompt: str, allow_tool_call: bool = True) -> AsyncIterator[str]:
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=self.queue_maxsize)
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                host = self.audio_manager.base_url or "http://127.0.0.1:11434"
                system_prompt = self.audio_manager.system_prompt
                if allow_tool_call:
                    system_prompt = f"{system_prompt}\n{TOOL_CALL_SYSTEM_PROMPT}"
                try:
                    ollama_module = importlib.import_module("ollama")
                except ImportError:
                    response = requests.post(
                        f"{host.rstrip('/')}/api/chat",
                        json={
                            "model": self.audio_manager.model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt},
                            ],
                            "stream": True,
                            "keep_alive": self.audio_manager.ollama_keep_alive,
                            "think": self.audio_manager.ollama_think,
                            "options": self.audio_manager._build_ollama_options(),
                        },
                        stream=True,
                        timeout=self.audio_manager.request_timeout_seconds,
                    )
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        event = json.loads(line)
                        message = event.get("message") or {}
                        tool_calls = self.audio_manager._extract_ollama_tool_calls(message)
                        if tool_calls:
                            for call in tool_calls:
                                function_info = call.get("function", {})
                                payload = {
                                    "tool_call": {
                                        "name": function_info.get("name", ""),
                                        "arguments": function_info.get("arguments", {}) or {},
                                    }
                                }
                                asyncio.run_coroutine_threadsafe(queue.put(json.dumps(payload, ensure_ascii=False)), loop).result()
                            continue
                        text = str(message.get("content", "") or "")
                        if text:
                            asyncio.run_coroutine_threadsafe(queue.put(text), loop).result()
                    return

                client = ollama_module.Client(host=host, timeout=self.audio_manager.request_timeout_seconds)
                stream = client.chat(
                    model=self.audio_manager.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                    keep_alive=self.audio_manager.ollama_keep_alive,
                    think=self.audio_manager.ollama_think,
                    options=self.audio_manager._build_ollama_options(),
                )
                for event in stream:
                    message = self.audio_manager._extract_ollama_message(event)
                    tool_calls = self.audio_manager._extract_ollama_tool_calls(message)
                    if tool_calls:
                        for call in tool_calls:
                            function_info = call.get("function", {})
                            payload = {
                                "tool_call": {
                                    "name": function_info.get("name", ""),
                                    "arguments": function_info.get("arguments", {}) or {},
                                }
                            }
                            asyncio.run_coroutine_threadsafe(queue.put(json.dumps(payload, ensure_ascii=False)), loop).result()
                        continue
                    text = str(message.get("content", "") or "")
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        producer_task = loop.run_in_executor(self._executor, _producer)
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            await producer_task

    def build_augmented_prompt(self, transcript: str) -> str:
        memory_context = self.audio_manager.conversation_memory.render()
        rag_context = self.rag_agent.build_prompt_context(
            transcript,
            k=2,
            max_context_chars=self.rag_context_chars,
        ) if self.enable_rag else ""
        sections = []
        if memory_context:
            sections.append(f"[recent_dialogue]\n{memory_context}")
        if rag_context:
            sections.append(rag_context)
        sections.append(f"[voice]\n{transcript}")
        sections.append(
            "규칙: 날씨 질문은 반드시 get_weather Tool Call JSON만 출력한다. "
            "일반 지식 질문은 [답변]/[검증] 형식으로 답한다. "
            "검색 컨텍스트는 필요한 경우에만 사용하고, 불필요한 사족 없이 한국어로 답한다."
        )
        return "\n\n".join(sections)

    async def execute_tool_call(self, tool_call: ToolCall) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.tool_agent.execute(tool_call.name, tool_call.arguments),
        )

    async def maybe_detect_barge_in(
        self,
        enabled: bool,
        stop_event: asyncio.Event,
        interrupt_queue: asyncio.Queue[str],
        timeout: float,
        phrase_time_limit: float,
    ) -> None:
        if not enabled:
            return
        while not stop_event.is_set():
            try:
                text = await self.transcribe_once(
                    input_type="mic",
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            except Exception:
                await asyncio.sleep(0.05)
                continue
            normalized = re.sub(r"\s+", " ", text or "").strip()
            if normalized:
                await interrupt_queue.put(normalized)
                stop_event.set()
                return

    async def enqueue_text_for_tts(
        self,
        text: str,
        tts_queue: asyncio.Queue[Optional[str]],
        pending_tts_text: str,
        emitted_tts_units: int,
    ) -> tuple[str, int]:
        pending_tts_text = f"{pending_tts_text} {text}".strip()
        min_chars = self.first_chunk_min_chars if emitted_tts_units == 0 else self.chunk_min_chars
        units, pending_tts_text = _split_tts_units(
            pending_tts_text,
            min_chars=min_chars,
            max_chars=self.chunk_max_chars,
        )
        for unit in units:
            emitted_tts_units += 1
            await tts_queue.put(unit)
        return pending_tts_text, emitted_tts_units

    async def tts_synthesizer(
        self,
        text_queue: asyncio.Queue[Optional[str]],
        playback_queue: asyncio.Queue[Optional[str]],
    ) -> list[str]:
        loop = asyncio.get_running_loop()
        output_dir = self.output_path.parent if self.output_path.parent != Path("") else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_paths: list[str] = []
        index = 0

        while True:
            chunk = await text_queue.get()
            try:
                if chunk is None:
                    break
                index += 1
                chunk_path = output_dir / f"{self.output_path.stem}_{index:03d}.mp3"
                synth_start = time.perf_counter()
                await loop.run_in_executor(
                    self._executor,
                    lambda text=chunk, path=chunk_path: self.audio_manager.synthesize_speech(text, str(path)),
                )
                print(f"[TTS 합성 지연] chunk={index} latency={time.perf_counter() - synth_start:.2f}s", flush=True)
                audio_paths.append(str(chunk_path))
                await playback_queue.put(str(chunk_path))
            finally:
                text_queue.task_done()

        await playback_queue.put(None)
        return audio_paths

    async def playback_consumer(self, playback_queue: asyncio.Queue[Optional[str]], stop_event: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        prebuffer: list[str] = []
        target_prebuffer = max(0, self.playback_prebuffer_chunks)

        while len(prebuffer) < target_prebuffer and not stop_event.is_set():
            item = await playback_queue.get()
            try:
                if item is None:
                    return
                prebuffer.append(item)
            finally:
                playback_queue.task_done()

        for audio_path in prebuffer:
            if stop_event.is_set():
                return
            await loop.run_in_executor(self._executor, lambda path=audio_path: _play_audio_file_blocking(path, stop_event))

        while not stop_event.is_set():
            item = await playback_queue.get()
            try:
                if item is None:
                    break
                await loop.run_in_executor(self._executor, lambda path=item: _play_audio_file_blocking(path, stop_event))
            finally:
                playback_queue.task_done()

    async def wait_for_tts_or_barge_in(
        self,
        tts_task: asyncio.Task[list[str]],
        playback_task: asyncio.Task[None],
        interrupt_queue: asyncio.Queue[str],
        playback_stop: asyncio.Event,
        tts_queue: asyncio.Queue[Optional[str]],
        playback_queue: asyncio.Queue[Optional[str]],
    ) -> tuple[list[str], str]:
        interrupted_text = ""
        while not playback_task.done():
            if not interrupt_queue.empty():
                interrupted_text = await interrupt_queue.get()
                playback_stop.set()
                _drain_async_queue(tts_queue)
                _drain_async_queue(playback_queue)
                self.audio_manager.halt_tts_playback()
                break
            await asyncio.sleep(0.05)

        if interrupted_text:
            for task in (tts_task, playback_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            return [], interrupted_text

        audio_paths = await tts_task
        await playback_task
        return audio_paths, ""

    async def run_once(
        self,
        input_type: str = "mic",
        file_path: Optional[str] = None,
        mic_timeout: float = DEFAULT_MIC_TIMEOUT,
        mic_phrase_time_limit: float = DEFAULT_MIC_PHRASE_TIME_LIMIT,
        enable_barge_in: bool = False,
        barge_in_timeout: float = 0.4,
        barge_in_phrase_time_limit: float = 2.5,
        initial_transcript: Optional[str] = None,
    ) -> dict:
        transcript = initial_transcript or await self.transcribe_once(
            input_type=input_type,
            file_path=file_path,
            timeout=mic_timeout,
            phrase_time_limit=mic_phrase_time_limit,
        )
        print(f"[STT 결과] : {transcript}", flush=True)
        if not transcript.strip():
            return {"transcribed_text": transcript, "llm_reply": "", "audio_paths": [], "state": "COMPLETED"}

        tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=self.queue_maxsize)
        playback_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=max(2, self.queue_maxsize // 2))
        playback_stop = asyncio.Event()
        tts_task = asyncio.create_task(self.tts_synthesizer(tts_queue, playback_queue))
        playback_task = asyncio.create_task(self.playback_consumer(playback_queue, playback_stop))
        interrupt_stop = asyncio.Event()
        interrupt_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        barge_in_task = asyncio.create_task(
            self.maybe_detect_barge_in(
                enable_barge_in,
                interrupt_stop,
                interrupt_queue,
                barge_in_timeout,
                barge_in_phrase_time_limit,
            )
        )
        llm_chunks: list[str] = []
        pending_tts_text = ""
        emitted_tts_units = 0
        tool_scan_buffer = ""
        tool_call: Optional[ToolCall] = None
        interrupted_text = ""

        try:
            prompt = self.build_augmented_prompt(transcript)
            async for chunk in self.stream_llm_raw(prompt, allow_tool_call=True):
                if not interrupt_queue.empty():
                    interrupted_text = await interrupt_queue.get()
                    break
                tool_scan_buffer = (tool_scan_buffer + chunk)[-2000:]
                detected_call = extract_tool_call_from_text(tool_scan_buffer)
                if detected_call is not None:
                    tool_call = detected_call
                    print(f"[TOOL CALL 감지] {tool_call.name} args={tool_call.arguments}", flush=True)
                    _drain_async_queue(tts_queue)
                    _drain_async_queue(playback_queue)
                    self.audio_manager.halt_tts_playback()
                    pending_tts_text = ""
                    break
                if "tool_call" in tool_scan_buffer and "}" not in tool_scan_buffer:
                    continue
                llm_chunks.append(chunk)
                pending_tts_text, emitted_tts_units = await self.enqueue_text_for_tts(
                    chunk,
                    tts_queue,
                    pending_tts_text,
                    emitted_tts_units,
                )

            if tool_call is not None:
                tool_result = await self.execute_tool_call(tool_call)
                print(f"[TOOL RESULT] {json.dumps(tool_result, ensure_ascii=False)}", flush=True)
                reinjected_prompt = (
                    f"{prompt}\n\n"
                    f"[tool_result]\n{json.dumps({'name': tool_call.name, 'result': tool_result}, ensure_ascii=False)}\n\n"
                    "도구 결과를 반영해서 JSON 없이 반드시 [답변]과 [검증] 섹션으로만 답해. "
                    "날씨 도구 결과에 error가 있으면 [답변]: 날씨 정보를 불러오는 데 실패했습니다. 통신 상태를 확인해주세요. "
                    "[검증]: 날씨 API 응답 실패가 확인되어 임의 날씨를 말하지 않았습니다. 라고 답해."
                )
                llm_chunks = []
                tool_scan_buffer = ""
                async for chunk in self.stream_llm_raw(reinjected_prompt, allow_tool_call=False):
                    if not interrupt_queue.empty():
                        interrupted_text = await interrupt_queue.get()
                        break
                    llm_chunks.append(chunk)
                    pending_tts_text, emitted_tts_units = await self.enqueue_text_for_tts(
                        chunk,
                        tts_queue,
                        pending_tts_text,
                        emitted_tts_units,
                    )

            tail_units, pending_tts_text = _split_tts_units(
                pending_tts_text,
                min_chars=self.chunk_min_chars,
                max_chars=self.chunk_max_chars,
                force=True,
            )
            for unit in tail_units:
                emitted_tts_units += 1
                await tts_queue.put(unit)
        finally:
            interrupt_stop.set()
            if interrupted_text:
                _drain_async_queue(tts_queue)
                _drain_async_queue(playback_queue)
                self.audio_manager.halt_tts_playback()
            await tts_queue.put(None)
            if not interrupted_text:
                audio_paths, interrupted_text = await self.wait_for_tts_or_barge_in(
                    tts_task,
                    playback_task,
                    interrupt_queue,
                    playback_stop,
                    tts_queue,
                    playback_queue,
                )
            else:
                playback_stop.set()
                for task in (tts_task, playback_task):
                    if not task.done():
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
            barge_in_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await barge_in_task
            gc.collect()

        if interrupted_text:
            print(f"[BARGE-IN] 새 발화 감지: {interrupted_text}", flush=True)
            return await self.run_once(
                input_type=input_type,
                file_path=file_path,
                mic_timeout=mic_timeout,
                mic_phrase_time_limit=mic_phrase_time_limit,
                enable_barge_in=enable_barge_in,
                barge_in_timeout=barge_in_timeout,
                barge_in_phrase_time_limit=barge_in_phrase_time_limit,
                initial_transcript=interrupted_text,
            )

        llm_reply = " ".join(llm_chunks).strip()
        self.audio_manager.conversation_memory.add_turn(transcript, llm_reply)
        print(f"[LLM 판단] : {llm_reply}", flush=True)
        return {
            "transcribed_text": transcript,
            "llm_reply": llm_reply,
            "audio_paths": audio_paths,
            "audio_path": audio_paths[0] if audio_paths else self.output_path,
            "state": "COMPLETED",
        }

    async def run_conversation(
        self,
        input_type: str = "mic",
        file_path: Optional[str] = None,
        mic_timeout: float = DEFAULT_MIC_TIMEOUT,
        mic_phrase_time_limit: float = DEFAULT_MIC_PHRASE_TIME_LIMIT,
        max_turns: int = 1,
        enable_barge_in: bool = False,
    ) -> dict:
        result: dict = {"state": "STOPPED"}
        for _ in range(max(1, max_turns)):
            result = await self.run_once(
                input_type=input_type,
                file_path=file_path,
                mic_timeout=mic_timeout,
                mic_phrase_time_limit=mic_phrase_time_limit,
                enable_barge_in=enable_barge_in,
            )
            if input_type == "file" or result.get("state") != "COMPLETED":
                break
            gc.collect()
        return result

    def close(self) -> None:
        self.audio_manager.close()
        self._executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="마이크 실시간 STT-LLM-TTS 파이프라인")
    parser.add_argument("-o", "--output", default="response.mp3", help="저장할 TTS 오디오 파일 경로")
    parser.add_argument(
        "--input-type",
        choices=["mic", "file"],
        default="mic",
        help="STT 입력 타입 선택 (mic 또는 file)",
    )
    parser.add_argument(
        "--file-path",
        default=None,
        help="input-type이 file일 때 사용할 오디오 파일 경로",
    )
    parser.add_argument(
        "--mic-timeout",
        type=float,
        default=DEFAULT_MIC_TIMEOUT,
        help="마이크 입력 시작 대기 시간(초)",
    )
    parser.add_argument(
        "--mic-phrase-time-limit",
        type=float,
        default=DEFAULT_MIC_PHRASE_TIME_LIMIT,
        help="한 번에 녹음할 최대 음성 길이(초)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="LLM 출력 청크를 TTS 큐로 즉시 보내는 저지연 async 파이프라인 사용",
    )
    parser.add_argument(
        "--queue-maxsize",
        type=int,
        default=DEFAULT_QUEUE_MAXSIZE,
        help="LLM-TTS 사이 async 큐 크기. TTFB가 길면 4~6, 끊김이 있으면 8~12 권장",
    )
    parser.add_argument(
        "--first-chunk-min-chars",
        type=int,
        default=DEFAULT_FIRST_CHUNK_MIN_CHARS,
        help="첫 TTS 청크 최소 글자 수. TTFB 단축은 8~12 권장",
    )
    parser.add_argument(
        "--chunk-min-chars",
        type=int,
        default=DEFAULT_CHUNK_MIN_CHARS,
        help="이후 TTS 청크 최소 글자 수. 끊김 완화는 24~40 권장",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=DEFAULT_CHUNK_MAX_CHARS,
        help="구두점이 없을 때 강제 분할할 최대 글자 수",
    )
    parser.add_argument(
        "--playback-prebuffer-chunks",
        type=int,
        default=DEFAULT_PLAYBACK_PREBUFFER_CHUNKS,
        help="재생 시작 전 준비할 TTS 오디오 청크 수. 끊김이 있으면 2 권장",
    )
    parser.add_argument("--disable-rag", action="store_true", help="로컬 RAG 컨텍스트 삽입 비활성화")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama 모델 이름. 기본값은 로컬 qwen2.5")
    parser.add_argument("--rag-corpus-path", default="./knowledge_base", help="로봇 매뉴얼/외부 지식 문서 디렉터리")
    parser.add_argument("--rag-context-chars", type=int, default=DEFAULT_RAG_CONTEXT_CHARS, help="RAG 컨텍스트 최대 문자 수")
    parser.add_argument("--enable-barge-in", action="store_true", help="TTS 재생 중 새 마이크 입력이 들어오면 현재 응답 중단")
    parser.add_argument("--max-turns", type=int, default=1, help="streaming 모드에서 처리할 연속 대화 턴 수")
    args = parser.parse_args()

    if args.input_type == "file" and not args.file_path:
        parser.error("--input-type file 사용 시 --file-path가 필요합니다.")

    if args.streaming:
        pipeline = StreamingVoicePipeline(
            output_path=args.output,
            model=args.model,
            queue_maxsize=args.queue_maxsize,
            first_chunk_min_chars=args.first_chunk_min_chars,
            chunk_min_chars=args.chunk_min_chars,
            chunk_max_chars=args.chunk_max_chars,
            playback_prebuffer_chunks=args.playback_prebuffer_chunks,
            rag_corpus_path=args.rag_corpus_path,
            rag_context_chars=args.rag_context_chars,
            enable_rag=not args.disable_rag,
        )
        try:
            asyncio.run(
                pipeline.run_conversation(
                    input_type=args.input_type,
                    file_path=args.file_path,
                    mic_timeout=args.mic_timeout,
                    mic_phrase_time_limit=args.mic_phrase_time_limit,
                    max_turns=args.max_turns,
                    enable_barge_in=args.enable_barge_in,
                )
            )
        finally:
            pipeline.close()
        raise SystemExit(0)

    run_pipeline(
        output_path=args.output,
        model=args.model,
        input_type=args.input_type,
        file_path=args.file_path,
        mic_timeout=args.mic_timeout,
        mic_phrase_time_limit=args.mic_phrase_time_limit,
    )
