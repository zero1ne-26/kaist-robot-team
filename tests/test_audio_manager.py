import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import audio_manager
import requests


class FakeOllamaResponse:
    def __init__(self, content: str, tool_calls=None):
        self.message = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        }


class FakeOllamaClient:
    response_text = "안녕하세요."
    response_sequence = []
    all_chat_kwargs = []
    last_chat_kwargs = {}
    last_host = None

    def __init__(self, host=None, **_kwargs):
        FakeOllamaClient.last_host = host

    def chat(self, **kwargs):
        FakeOllamaClient.last_chat_kwargs = kwargs
        FakeOllamaClient.all_chat_kwargs.append(kwargs)
        stream = bool(kwargs.get("stream", False))

        if FakeOllamaClient.response_sequence:
            item = FakeOllamaClient.response_sequence.pop(0)

            if stream:
                chunks = item.get("stream_chunks")
                if chunks is None:
                    chunks = [item.get("content", "")]

                def _iter_chunks():
                    for chunk_text in chunks:
                        yield FakeOllamaResponse(content=str(chunk_text), tool_calls=[])

                return _iter_chunks()

            return FakeOllamaResponse(
                content=item.get("content", ""),
                tool_calls=item.get("tool_calls", []),
            )

        if stream:
            def _iter_default():
                yield FakeOllamaResponse(FakeOllamaClient.response_text)

            return _iter_default()

        return FakeOllamaResponse(FakeOllamaClient.response_text)


class FakeOllamaModule:
    Client = FakeOllamaClient


class AudioManagerStreamingTest(unittest.TestCase):
    def test_stream_remote_llm_collects_reasoning_text(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "안녕하세요."
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("안녕"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(chunks, ["안녕하세요."])

    def test_stream_remote_llm_filters_reasoning_sentences(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "여기서는 생각 과정을 적겠습니다. 안녕하세요. 저는 자비스입니다."
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("안녕"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(chunks, ["안녕하세요.", "저는 자비스입니다."])

    def test_default_system_prompt_guides_speech_only_output(self):
        manager = audio_manager.AudioManager(model="demo-model")

        self.assertIn("한국어 음성비서", manager.system_prompt)
        self.assertIn("사용자의 질문에 직접 답해", manager.system_prompt)
        self.assertIn("함수 호출 JSON을 만들지 마", manager.system_prompt)
        self.assertIn("미래 예언, 존재하지 않는 사실", manager.system_prompt)
        self.assertIn("헤헤, 그건 지니가 아직 공부를 못해서 잘 몰라", manager.system_prompt)

    def test_voice_prompt_classifies_common_question_types(self):
        manager = audio_manager.AudioManager(model="demo-model")

        lion_prompt = manager._normalize_prompt_for_llm("[voice]\n진이야. 왜 사자는 물이 생활을 해?")
        company_prompt = manager._normalize_prompt_for_llm("[voice]\n현대자동차 그룹이 뭐야?")
        lunch_prompt = manager._normalize_prompt_for_llm("[voice]\n오늘 점심메뉴는 뭐 먹을까?")

        self.assertIn("무리 생활", lion_prompt)
        self.assertIn("질문의 전제를 함부로 부정하지 말고", lion_prompt)
        self.assertIn("기업, 기관, 브랜드에 대한 일반 지식 질문", company_prompt)
        self.assertIn("일상 추천 요청", lunch_prompt)

    def test_tts_sanitizer_removes_non_korean_cjk_fragments(self):
        manager = audio_manager.AudioManager(model="demo-model")

        self.assertNotIn("什么", manager._sanitize_for_tts("오늘 점심 먹什么呢? 김치찌개를 추천해요."))
        self.assertNotIn("hee", manager._sanitize_for_tts("헤hee, 블랙홀은 중력이 강한 곳이야!"))

    def test_default_model_settings_are_used(self):
        manager = audio_manager.AudioManager()
        self.assertEqual(manager.model, "exaone3.5:2.4b")
        self.assertEqual(manager.vlm_model, "qwen2.5vl:3b")
        self.assertGreaterEqual(manager.vlm_timeout_seconds, manager.request_timeout_seconds)

    def test_exaone_uses_builtin_tool_intent_without_ollama_tools(self):
        manager = audio_manager.AudioManager(model="exaone3.5:2.4b")

        self.assertFalse(manager._model_supports_ollama_tools())
        self.assertIn("알람 설정 완료", manager._handle_builtin_tool_intent("내일 아침 7시에 알람 맞춰줘"))
        self.assertIn("장치 제어 모듈", manager._handle_builtin_tool_intent("블루투스 스피커 켜줘"))

    def test_builtin_weather_intent_uses_spoken_location(self):
        manager = audio_manager.AudioManager(model="exaone3.5:2.4b")

        with patch.object(audio_manager, "get_weather", return_value={"weather_ko": "맑음", "temperature_c": "24"}) as mocked_weather:
            answer = manager._handle_builtin_tool_intent("서울 날씨 어때?")

        mocked_weather.assert_called_once_with("Seoul")
        self.assertIn("서울", answer)
        self.assertIn("24도씨", answer)

    def test_stream_remote_llm_emits_debug_message_for_text(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "안녕하세요."
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            with patch("builtins.print") as mocked_print:
                chunks = list(manager.stream_remote_llm("안녕"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(chunks, ["안녕하세요."])
        self.assertTrue(any("[DEBUG] 전달할 LLM 텍스트:" in str(call.args[0]) for call in mocked_print.call_args_list))

    def test_stream_remote_llm_uses_system_instruction_and_tools(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "안녕하세요."
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="qwen2.5")
            list(manager.stream_remote_llm("대전 날씨 알려줘"))
        finally:
            audio_manager.ollama = original_ollama

        first_call = FakeOllamaClient.all_chat_kwargs[0]
        self.assertEqual(first_call.get("model"), "qwen2.5")
        self.assertEqual(first_call.get("keep_alive"), "5m")
        first_messages = first_call.get("messages", [])
        self.assertTrue(first_messages)
        self.assertEqual(first_messages[0].get("role"), "system")
        self.assertIn("한국어 음성비서", first_messages[0].get("content", ""))
        self.assertIn("질문 유형", first_messages[0].get("content", ""))

        tools = first_call.get("tools", [])
        tool_names = [tool.get("function", {}).get("name", "") for tool in tools]
        self.assertIn("get_weather", tool_names)
        self.assertIn("search_web", tool_names)
        self.assertIn("set_alarm", tool_names)
        self.assertIn("send_message", tool_names)

    def test_stream_remote_llm_routes_search_web_tool(self):
        original_ollama = audio_manager.ollama
        original_search_web = audio_manager.search_web
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_web",
                            "arguments": {"query": "오바마 대통령의 2026년 최신 근황"},
                        }
                    }
                ],
            },
            {
                "content": "오바마 대통령의 최신 근황을 검색해서 알려드릴게요.",
            },
            {
                "stream_chunks": ["오바마 대통령의 최신 근황을 검색해서 알려드릴게요."],
            },
        ]
        FakeOllamaClient.all_chat_kwargs = []
        audio_manager.search_web = lambda query="": {"query": query, "result": f"웹 검색 결과: {query}"}

        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("오바마 대통령의 2026년 최신 근황 알려줘"))
        finally:
            audio_manager.ollama = original_ollama
            audio_manager.search_web = original_search_web

        self.assertIn("search_web", [tool.get("function", {}).get("name", "") for tool in FakeOllamaClient.all_chat_kwargs[0].get("tools", [])])
        self.assertTrue(chunks)

    def test_stream_remote_llm_routes_set_alarm_tool(self):
        original_ollama = audio_manager.ollama
        original_set_alarm = audio_manager.set_alarm
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "set_alarm",
                            "arguments": {"time": "1시간 뒤"},
                        }
                    }
                ],
            },
            {
                "content": "1시간 뒤 알람을 설정했어요.",
            },
            {
                "stream_chunks": ["1시간 뒤 알람을 설정했어요."],
            },
        ]
        FakeOllamaClient.all_chat_kwargs = []
        audio_manager.set_alarm = lambda time_value="": {"time": time_value, "result": f"알람 설정 완료: {time_value}"}

        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("1시간 뒤에 알람 맞춰줘"))
        finally:
            audio_manager.ollama = original_ollama
            audio_manager.set_alarm = original_set_alarm

        self.assertIn("set_alarm", [tool.get("function", {}).get("name", "") for tool in FakeOllamaClient.all_chat_kwargs[0].get("tools", [])])
        self.assertTrue(chunks)

    def test_stream_remote_llm_refuses_unknown_prompt(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []

        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("지구에서 100만 광년 떨어진 외계인의 이름은 뭐야?"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(FakeOllamaClient.all_chat_kwargs, [])
        self.assertTrue(chunks)
        self.assertIn("지니가 아직 공부를 못해서 잘 몰라", chunks[0])

    def test_stream_remote_llm_uses_direct_route_for_general_questions(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "우주는 정말 커요. 로켓은 연료를 분사해 위로 올라가요. 또 궁금한 게 있으면 언제든지 지니에게 물어봐!"
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("우주선은 어떻게 우주로 날아가?"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(len(FakeOllamaClient.all_chat_kwargs), 1)
        self.assertNotIn("tools", FakeOllamaClient.all_chat_kwargs[0])
        self.assertTrue(chunks)
        self.assertLessEqual(len(chunks), manager.max_tts_sentences)
        self.assertIn("우주", " ".join(chunks))

    def test_stream_remote_llm_handles_cumulative_chunks_without_duplication(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {"stream_chunks": ["안", "안녕", "안녕하세요.", "안녕하세요. 반", "안녕하세요. 반가워요."]},
        ]
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("인사해줘"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(chunks, ["안녕하세요.", "반가워요."])

    def test_stream_remote_llm_deduplicates_consecutive_identical_sentences(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {"stream_chunks": ["안녕하세요.", "안녕하세요.", "반갑습니다."]},
        ]
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("인사해줘"))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(chunks, ["안녕하세요.", "반갑습니다."])

    def test_stream_remote_llm_executes_tool_call_and_sends_back_result(self):
        original_ollama = audio_manager.ollama
        original_weather_func = audio_manager.get_current_weather
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "Daejeon"},
                        }
                    }
                ],
            },
            {
                "content": "현재 계신 대전의 날씨는 맑으며, 기온은 25도입니다.",
            },
            {
                "stream_chunks": ["현재 계신 대전의 날씨는 맑으며, 기온은 25도입니다."],
            },
        ]
        FakeOllamaClient.all_chat_kwargs = []
        audio_manager.get_current_weather = lambda location="Daejeon": {
            "location": location,
            "temperature_c": "25",
            "weather": "맑음",
        }

        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("대전 날씨 알려줘"))
        finally:
            audio_manager.ollama = original_ollama
            audio_manager.get_current_weather = original_weather_func

        self.assertEqual(chunks, ["현재 계신 대전의 날씨는 맑으며, 기온은 25도입니다."])
        self.assertEqual(len(FakeOllamaClient.all_chat_kwargs), 3)
        second_messages = FakeOllamaClient.all_chat_kwargs[1].get("messages", [])
        tool_messages = [msg for msg in second_messages if msg.get("role") == "tool"]
        self.assertTrue(tool_messages)

    def test_stream_remote_llm_parses_malformed_tool_json_text(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {
                "content": "tool_calls: {\"function\": {\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Daejeon\"}}}",
            },
            {
                "content": "안녕하세요.",
            },
            {
                "stream_chunks": ["안녕하세요."],
            },
        ]
        FakeOllamaClient.all_chat_kwargs = []
        original_weather_func = audio_manager.get_current_weather
        audio_manager.get_current_weather = lambda location="Daejeon": {
            "location": location,
            "temperature_c": "25",
            "weather": "맑음",
        }

        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("대전 날씨 알려줘"))
        finally:
            audio_manager.ollama = original_ollama
            audio_manager.get_current_weather = original_weather_func

        self.assertTrue(chunks)
        self.assertIn("안녕하세요.", chunks[-1])

    def test_sanitize_for_tts_removes_markdown_and_emojis(self):
        manager = audio_manager.AudioManager(model="demo-model")

        sanitized = manager._sanitize_for_tts("*안녕* 😊 마침표 #테스트")

        self.assertNotIn("*", sanitized)
        self.assertNotIn("😊", sanitized)
        self.assertNotIn("마침표", sanitized)
        self.assertIn("안녕", sanitized)

    def test_sanitize_for_tts_removes_fahrenheit_mentions(self):
        manager = audio_manager.AudioManager(model="demo-model")

        sanitized = manager._sanitize_for_tts("기온은 24°C 76°F, 화씨 기준으로는 76F입니다.")

        self.assertNotIn("76°F", sanitized)
        self.assertNotIn("화씨", sanitized)
        self.assertIn("24°C", sanitized)

    def test_stream_remote_llm_handles_repeated_tool_call_steps(self):
        original_ollama = audio_manager.ollama
        original_weather_func = audio_manager.get_current_weather
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_sequence = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {},
                        }
                    }
                ],
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "Daejeon"},
                        }
                    }
                ],
            },
            {
                "content": "오늘 대전은 맑고 기온은 25도입니다.",
            },
            {
                "stream_chunks": ["오늘 대전은 맑고 기온은 25도입니다."],
            },
        ]
        FakeOllamaClient.all_chat_kwargs = []
        audio_manager.get_current_weather = lambda location="Daejeon": {
            "location": location,
            "temperature_c": "25",
            "weather": "맑음",
        }

        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_llm("오늘 날씨 어때?"))
        finally:
            audio_manager.ollama = original_ollama
            audio_manager.get_current_weather = original_weather_func

        self.assertEqual(chunks, ["오늘 대전은 맑고 기온은 25도입니다."])
        self.assertEqual(len(FakeOllamaClient.all_chat_kwargs), 4)

    def test_stream_remote_vlm_accepts_multimodal_image_inputs(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "확인했습니다."
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            chunks = list(manager.stream_remote_vlm("이미지 설명해줘", image_inputs=[b"fake-image-bytes"]))
        finally:
            audio_manager.ollama = original_ollama

        self.assertEqual(chunks, ["확인했습니다."])
        first_call = FakeOllamaClient.all_chat_kwargs[0]
        self.assertEqual(first_call.get("model"), "qwen2.5vl:3b")
        user_message = first_call.get("messages", [])[1]
        self.assertEqual(user_message.get("role"), "user")
        self.assertTrue(user_message.get("images"))

    def test_get_current_weather_timeout_returns_error_dict(self):
        with patch("audio_manager.requests.get", side_effect=requests.exceptions.Timeout("timeout")):
            result = audio_manager.get_current_weather("Daejeon")

        self.assertEqual(result.get("error"), "timeout")
        self.assertEqual(result.get("message"), "기상청 서버 응답 지연")

    def test_get_current_weather_success_returns_json_dict(self):
        mocked_response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "current_condition": [{
                    "temp_C": "25",
                    "weatherDesc": [{"value": "Sunny"}],
                }],
                "nearest_area": [{
                    "areaName": [{"value": "Daejeon"}],
                }],
            },
        )
        with patch("audio_manager.requests.get", return_value=mocked_response):
            result = audio_manager.get_current_weather("Daejeon")

        self.assertEqual(result.get("location"), "Daejeon")
        self.assertEqual(result.get("temperature_c"), "25")
        self.assertEqual(result.get("weather"), "Sunny")
        self.assertIn("현재 Daejeon의 날씨는", result["summary_ko"])


    def test_get_current_weather_uses_daejeon_when_location_missing(self):
        mocked_response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "current_condition": [{
                    "temp_C": "25",
                    "weatherDesc": [{"value": "Sunny"}],
                }],
                "nearest_area": [{
                    "areaName": [{"value": "Daejeon"}],
                }],
            },
        )
        with patch("audio_manager.requests.get", return_value=mocked_response) as mocked_get:
            result = audio_manager.get_current_weather("")

        self.assertIn("/Daejeon?format=j1", mocked_get.call_args.args[0])
        self.assertEqual(result.get("location"), "Daejeon")

    def test_clean_llm_text_removes_reasoning_markers_and_symbols(self):
        manager = audio_manager.AudioManager(model="demo-model")
        cleaned = manager._clean_llm_text("생각 과정: 먼저 답을 구성하겠습니다. **답변:** 안녕하세요, 저는 자비스입니다!")
        self.assertEqual(cleaned, "안녕하세요, 저는 자비스입니다!")

    def test_clean_llm_text_preserves_leading_spoken_numbers(self):
        manager = audio_manager.AudioManager(model="demo-model")

        cleaned = manager._clean_llm_text("1시간 뒤 알람을 설정했어요.")

        self.assertEqual(cleaned, "1시간 뒤 알람을 설정했어요.")

    def test_stream_remote_llm_normalizes_internal_voice_context(self):
        original_ollama = audio_manager.ollama
        audio_manager.ollama = FakeOllamaModule
        FakeOllamaClient.response_text = "안녕하세요."
        FakeOllamaClient.response_sequence = []
        FakeOllamaClient.all_chat_kwargs = []
        try:
            manager = audio_manager.AudioManager(model="demo-model")
            list(manager.stream_remote_llm("[voice]\n안녕\n\n[visual_context]\npresence=True, face_expression=neutral"))
        finally:
            audio_manager.ollama = original_ollama

        user_message = FakeOllamaClient.all_chat_kwargs[0].get("messages", [])[1]
        self.assertNotIn("[voice]", user_message.get("content", ""))
        self.assertNotIn("[visual_context]", user_message.get("content", ""))
        self.assertEqual("안녕", user_message.get("content", ""))
        system_message = FakeOllamaClient.all_chat_kwargs[0].get("messages", [])[0]
        self.assertIn("질문 유형", system_message.get("content", ""))

    def test_tts_worker_synthesizes_single_final_file(self):
        manager = audio_manager.AudioManager(model="demo-model")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = audio_manager.Path(temp_dir)
            with patch.object(manager, "synthesize_speech", return_value=str(output_dir / "response.mp3")) as mocked_synthesize:
                with patch.object(manager, "_play_audio_file") as mocked_play:
                    manager.start_tts_worker(output_dir, "response")
                    manager.enqueue_sentence("안녕하세요.")
                    manager.enqueue_sentence("1시간 뒤 알람을 설정했어요.")
                    manager.stop_tts_worker()

        mocked_synthesize.assert_called_once()
        synthesized_text = mocked_synthesize.call_args.args[0]
        self.assertIn("안녕하세요.", synthesized_text)
        self.assertIn("1시간 뒤 알람을 설정했어요.", synthesized_text)
        mocked_play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
