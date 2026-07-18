# DEBUG REPORT

## Applied Fixes

- Hardened `audio_manager.py` streaming with outer exception guards and `logger.exception(...)` traceback output so Ollama/tool failures do not fail silently.
- Added regex-based JSON extraction fallback for malformed tool-call payloads and treated parse failures as a normal dialogue fallback path.
- Kept Ollama sessions warm with `keep_alive="5m"` on every chat call.
- Reduced Ollama context and decode budget for edge hardware: `num_ctx=1024`, `num_predict=48`, `temperature=0.0`.
- Added request timeouts for external weather calls: `timeout=3.0`.
- Added stream sanitization for TTS so markdown markers, emoji, and meta words such as `마침표` are stripped before synthesis.
- Added graceful shutdown handling for `Ctrl+C` and exit keywords in the state machine.

## Validation

- Unit tests: `17 passed`
- Runtime pipeline: `stt_tts_pipeline.py --input-type file --file-path "input.wave.m4a"`

## TTFT Measurements

- First run: `14.68s`
- Second run: `8.49s`

## Notes

- The pipeline now logs traceback details on exceptions instead of exiting silently.
- Keep-alive reduced the warm-run latency, but the tool-call path still dominates first-token latency on this hardware.