# Edge AI Architecture Report

## 1) Syntax Error Resolution

- File: audio_manager.py
- Issue: Unexpected indentation around the TTS worker shutdown block.
- Fix:
  - Corrected indentation in `stop_tts_worker()`.
  - Ensured cleanup/memory maintenance calls are aligned with the function scope.
- Validation:
  - Pylance/compile checks report no syntax errors.

## 2) Function Calling Structure (Zero-Credit Local Pipeline)

### Prompt-level contract
- The system prompt now enforces:
  - If a query needs external data/tool use, return tool invocation details in JSON form.
  - If the query is out of scope or unsolved by tools, never hallucinate and respond with:
    - "헤헤, 그건 지니가 아직 공부를 못해서 잘 몰라!"

### Runtime tool-calling loop
- Implemented in `stream_remote_llm()` and `_resolve_tool_calls()`:
  1. Build messages and optional tool schema.
  2. Ask local LLM for response.
  3. If tool calls are present, parse JSON arguments.
  4. Execute backend tool functions in Python (`get_current_weather`, `search_web`, `set_alarm`, `send_message`, etc.).
  5. Inject tool results back as tool-role context.
  6. Request final natural-language streaming response.

### OOD safeguard
- `_should_refuse_prompt()` catches clear OOD patterns and immediately returns:
  - "헤헤, 그건 지니가 아직 공부를 못해서 잘 몰라!"

## 3) TTS Smoothing and In-Memory Audio Consolidation

### Sentence-level buffering
- LLM stream deltas are accumulated and emitted by sentence boundaries (`.`, `!`, `?`, `\n`) with adaptive comma splitting for long segments.
- This prevents ultra-fragmented token-level TTS inputs.

### In-memory TTS pipeline
- Replaced per-sentence disk writes (`*_00.mp3`, `*_01.mp3`, ...) with RAM aggregation:
  - `synthesize_speech_bytes()` produces MP3 bytes in memory via `io.BytesIO`.
  - `_tts_worker()` appends sentence audio bytes to a shared in-memory buffer.
  - `stop_tts_worker()` writes a single final file (`response.mp3`) once at end of stream.

### Result
- No chunk mp3 files are generated during normal run.
- Only one final response file is persisted.

## 4) Memory Control Strategy for 8GB Edge Device

- Added explicit maintenance routine:
  - `run_memory_maintenance(stage)` performs:
    - `gc.collect()`
    - `torch.cuda.empty_cache()` when CUDA is available
- Maintenance hooks are called at key stage boundaries:
  - after STT (`post_stt`)
  - before and after LLM streaming (`pre_llm`, `post_llm`)
  - after each TTS sentence synthesis (`post_tts_sentence`)
  - after final TTS flush (`post_tts`)
- Existing 4-bit quantization + SDPA path remains in local LoRA loading:
  - `load_in_4bit=True`
  - `bnb_4bit_compute_dtype=torch.float16`
  - `bnb_4bit_use_double_quant=True`
  - `attn_implementation="sdpa"`

## 5) Verification Summary

- Syntax/compile: passed.
- Unit tests: passed (`tests/test_audio_manager.py` all green).
- End-to-end run (`input.wave.m4a`): completed with STT -> LLM(tool) -> TTS.
- File scan: only `response.mp3` output retained for the response path; chunk files (`*_0?.mp3`) not generated.
