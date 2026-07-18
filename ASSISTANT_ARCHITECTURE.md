# Assistant Architecture

## System Prompt

The assistant is now configured as a friendly Genie persona named `지니`.

Core rules:

- Start with a bright reaction such as `안녕! 정말 재미있는 질문이네!` or `우와, 아주 좋은 질문이야!`
- Explain simply with vivid analogies.
- Never use emoji, markdown markers, or special TTS-hostile symbols in the spoken output.
- End warmly with a line like `또 궁금한 게 있으면 언제든지 지니에게 물어봐!`
- Use `Daejeon` as the default location for weather questions.
- Use Celsius only. Do not mention Fahrenheit.
- First decide whether the question needs a tool, is answerable from common knowledge, or is unknowable.
- If it is unknowable, do not hallucinate. Reply with:
  `헤헤 미안해 그건 지니가 아직 공부를 못해서 잘 몰라 다른 재미있는 걸 물어볼래`

## Tool Registry

The assistant now supports four tool families.

### `get_current_weather(location)`

- Returns local weather information.
- Default location: `Daejeon`
- Output is localized with a Korean summary and Celsius only.

### `search_web(query)`

- Dummy latest-info lookup.
- Returns: `웹 검색 결과: {query}`

### `set_alarm(time)`

- Dummy alarm/timer setup.
- Returns: `알람 설정 완료: {time}`

### `send_message(contact, message)`

- Dummy phone/message control.
- Returns: `메시지 전송 완료: {contact}`

## Routing Policy

- Weather and device-control questions go through tool calling.
- General knowledge questions go through the direct streaming route with no tool delay.
- Clearly fabricated or unknowable questions are rejected immediately with the refusal phrase above.

## Validation

- Unit tests: `22 passed`
- Runtime direct question: `우주선은 어떻게 우주로 날아가?`
- Direct route TTFT observed: `5.38s`
- Output started with a bright Korean response and streamed without tool usage.

## Jetson Fine-Tuning

- Dataset generation is handled by `generate_dataset.py`, which produces JSONL behavior-cloning data.
- QLoRA training is handled by `train_qlora.py` with 4-bit NF4 quantization, SDPA attention, gradient checkpointing, and paged AdamW 8-bit.
- Optional LoRA inference loading is available in `audio_manager.py` so the runtime can keep using the Ollama path unless an adapter is explicitly enabled.
- The current workspace does not include the full training stack yet, so the training script is help-safe and import-safe, but actual training still requires `torch`, `transformers`, `datasets`, `peft`, and `trl`.