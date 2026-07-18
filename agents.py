from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from audio_manager import control_iot_device, get_current_weather, get_weather, search_web, send_message, set_alarm

logger = logging.getLogger(__name__)


TOOL_CALL_SYSTEM_PROMPT = """
너는 카이스트 로봇 연구소의 로봇 음성 상호작용용 AI 어시스턴트이다.
먼저 사용자 의도를 정보 검색, 일반 지식, 하드웨어 제어 중 하나로 구분한다.
날씨를 묻는 요청은 절대 추측하지 말고 반드시 get_weather 도구를 호출한다.
도구가 필요한 요청이면 반드시 순수 JSON 객체 한 개만 출력한다.
마크다운 코드블록, 설명 문장, 접두사, 접미사를 절대 붙이지 않는다.
JSON 스키마는 정확히 다음 중 하나를 사용한다.

단일 도구 호출:
{"tool_call":{"name":"도구명","arguments":{}}}

Ollama/Qwen 호환 함수 호출:
{"tool_calls":[{"function":{"name":"도구명","arguments":{}}}]}

사용 가능한 도구와 arguments:
- get_weather: {}
- check_robot_battery: {}
- control_quadruped_motor: {"action":"stand|sit|walk|stop|turn_left|turn_right","speed":"slow|normal|fast","duration_seconds":1.0}
- get_current_weather: {"location":"Daejeon"}
- search_web: {"query":"검색어"}
- set_alarm: {"time":"1시간 뒤"}
- send_message: {"contact":"이름","message":"내용"}
- control_iot_device: {"device_id":"bluetooth|light|plug","action":"on|off|dim"}

예시:
사용자: 오늘 날씨 어때?
출력: {"tool_call":{"name":"get_weather","arguments":{}}}

사용자: 배터리 얼마나 남았어?
출력: {"tool_call":{"name":"check_robot_battery","arguments":{}}}

사용자: 멈춰.
출력: {"tool_call":{"name":"control_quadruped_motor","arguments":{"action":"stop","speed":"normal","duration_seconds":1.0}}}

도구가 필요 없는 일반 지식 질문이면 JSON을 출력하지 말고 다음 형식으로만 출력한다.
[답변]: 핵심 내용을 3문장 이내로 정확하게 설명한다.
[검증]: 답변이 과학적, 역사적, 사실적으로 맞는지 스스로 검토하고 오류 여부를 명시한다.
도구 결과를 받은 뒤에는 JSON 없이 반드시 [답변]과 [검증] 섹션으로 최종 답변만 출력한다.
날씨 도구 결과에 {"error":"weather_service_timeout"}가 있으면 반드시 다음과 같이 출력한다.
[답변]: 날씨 정보를 불러오는 데 실패했습니다. 통신 상태를 확인해주세요.
[검증]: 날씨 API 응답 실패가 확인되어 임의 날씨를 말하지 않았습니다.
"네 알겠습니다", "제가 도와드릴게요" 같은 사족은 금지한다.
모든 출력은 한국어로 한다.
""".strip()


class HashEmbeddingFunction:
    """ChromaDB용 외부 모델 없는 고정 차원 해시 임베딩."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 - Chroma expects this parameter name.
        vectors: List[List[float]] = []
        for text in input:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"[A-Za-z0-9가-힣]+", (text or "").lower())
            for token in tokens:
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class AgentKind(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RAG = "rag"
    TOOL = "tool"


@dataclass(slots=True)
class Handoff:
    target: AgentKind
    reason: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResponse:
    agent: AgentKind
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps({"tool_call": {"name": self.name, "arguments": self.arguments}}, ensure_ascii=False)


def check_robot_battery() -> Dict[str, Any]:
    return {
        "status": "ok",
        "battery_percent": 87,
        "voltage": 15.8,
        "is_charging": False,
        "summary_ko": "현재 배터리는 87퍼센트이고 충전 중은 아닙니다.",
    }


def control_quadruped_motor(action: str, speed: str = "normal", duration_seconds: float = 1.0) -> Dict[str, Any]:
    normalized_action = (action or "").strip().lower()
    if normalized_action not in {"stand", "sit", "walk", "stop", "turn_left", "turn_right"}:
        return {"error": "invalid_action", "message": "action은 stand/sit/walk/stop/turn_left/turn_right 중 하나여야 합니다."}
    return {
        "status": "ok",
        "action": normalized_action,
        "speed": speed,
        "duration_seconds": max(0.0, float(duration_seconds or 0.0)),
        "summary_ko": f"사족보행 모터 명령 {normalized_action}을 실행했습니다.",
    }


def extract_tool_call_from_text(text: str) -> Optional[ToolCall]:
    candidate = (text or "").strip()
    if not candidate:
        return None
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list) and payload["tool_calls"]:
        first_call = payload["tool_calls"][0]
        if isinstance(first_call, dict):
            function_info = first_call.get("function") or first_call
            if isinstance(function_info, dict):
                payload = {
                    "tool_call": {
                        "name": function_info.get("name"),
                        "arguments": function_info.get("arguments") or {},
                    }
                }
    call = payload.get("tool_call") or payload.get("function") or payload
    if not isinstance(call, dict):
        return None
    name = str(call.get("name") or call.get("tool_name") or "").strip()
    arguments = call.get("arguments") or call.get("args") or {}
    if not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(name=name, arguments=arguments)


class BaseAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def handle(self, request: str, **kwargs: Any) -> AgentResponse:
        raise NotImplementedError


class ToolAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("tool")
        self._dispatch: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_weather": get_weather,
            "get_current_weather": get_current_weather,
            "search_web": search_web,
            "set_alarm": set_alarm,
            "send_message": send_message,
            "control_iot_device": control_iot_device,
            "check_robot_battery": check_robot_battery,
            "control_quadruped_motor": control_quadruped_motor,
        }

    def execute(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        arguments = arguments or {}
        handler = self._dispatch.get(tool_name)
        if handler is None:
            return {"error": "unknown_tool", "tool": tool_name}

        normalized_arguments = dict(arguments)
        if tool_name == "set_alarm":
            normalized_arguments = {"time_value": arguments.get("time") or arguments.get("time_value") or arguments.get("value")}
        elif tool_name == "control_iot_device":
            normalized_arguments = {
                "device_id": arguments.get("device_id") or arguments.get("device") or arguments.get("target"),
                "action": arguments.get("action") or arguments.get("state") or arguments.get("command"),
            }
        elif tool_name == "send_message":
            normalized_arguments = {
                "contact": arguments.get("contact") or arguments.get("target") or arguments.get("name"),
                "message": arguments.get("message") or arguments.get("text") or arguments.get("content"),
            }
        elif tool_name == "search_web":
            normalized_arguments = {"query": arguments.get("query") or arguments.get("text") or arguments.get("prompt")}
        elif tool_name in {"get_weather", "get_current_weather"}:
            normalized_arguments = {"location": arguments.get("location") or arguments.get("city") or arguments.get("place")}
        elif tool_name == "control_quadruped_motor":
            normalized_arguments = {
                "action": arguments.get("action") or arguments.get("command") or arguments.get("pose"),
                "speed": arguments.get("speed") or "normal",
                "duration_seconds": arguments.get("duration_seconds") or arguments.get("duration") or 1.0,
            }

        try:
            return handler(**normalized_arguments)
        except TypeError:
            return {"error": "bad_arguments", "tool": tool_name, "arguments": normalized_arguments}
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("tool execution failed: %s", tool_name)
            return {"error": "tool_execution_failed", "tool": tool_name, "message": str(exc)}

    def handle(self, request: str, **kwargs: Any) -> AgentResponse:
        tool_name = str(kwargs.get("tool_name", "")).strip()
        arguments = kwargs.get("arguments") or {}
        result = self.execute(tool_name, arguments)
        return AgentResponse(agent=AgentKind.TOOL, content=json.dumps(result, ensure_ascii=False), metadata={"tool_name": tool_name})


class RAGAgent(BaseAgent):
    def __init__(self, corpus_path: str = "./knowledge_base", max_context_chars: int = 900) -> None:
        super().__init__("rag")
        self.corpus_path = Path(corpus_path)
        self._documents: List[Dict[str, str]] = []
        self.max_context_chars = max(120, max_context_chars)
        self._chromadb = None
        self._collection = None
        self._load_backend()
        self.load_local_documents()

    def _load_backend(self) -> None:
        try:
            import chromadb  # type: ignore

            self._chromadb = chromadb
            client = chromadb.PersistentClient(path=str(self.corpus_path))
            self._collection = client.get_or_create_collection(
                "knowledge_base",
                embedding_function=HashEmbeddingFunction(),
            )
        except Exception:
            self._chromadb = None
            self._collection = None

    def ingest(self, documents: Iterable[Dict[str, str]]) -> None:
        docs = list(documents)
        self._documents.extend(docs)
        if self._collection is None:
            return

        ids = [doc.get("id") or f"doc-{index}" for index, doc in enumerate(docs, start=1)]
        texts = [doc.get("text", "") for doc in docs]
        metadatas = [{"source": doc.get("source", "local")} for doc in docs]
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def load_local_documents(self) -> None:
        if not self.corpus_path.exists():
            return
        docs = []
        for path in self.corpus_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for index, chunk in enumerate(self._chunk_text(text), start=1):
                docs.append({"id": f"{path.name}-{index}", "text": chunk, "source": str(path)})
        if docs:
            self.ingest(docs)

    def _chunk_text(self, text: str, chunk_chars: int = 700, overlap_chars: int = 80) -> List[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        chunks: List[str] = []
        cursor = 0
        while cursor < len(normalized):
            chunk = normalized[cursor:cursor + chunk_chars].strip()
            if chunk:
                chunks.append(chunk)
            cursor += max(1, chunk_chars - overlap_chars)
        return chunks

    def _lexical_score(self, query: str, text: str) -> float:
        query_tokens = re.findall(r"[A-Za-z0-9가-힣]+", query.lower())
        text_tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
        if not query_tokens or not text_tokens:
            return 0.0
        text_counts: Dict[str, int] = {}
        for token in text_tokens:
            text_counts[token] = text_counts.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            if token in text_counts:
                score += 1.0 + math.log1p(text_counts[token])
        return score

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        if self._collection is not None:
            result = self._collection.query(query_texts=[normalized_query], n_results=k)
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            return [
                {"text": text, "metadata": metadata or {}}
                for text, metadata in zip(documents, metadatas)
            ]

        lowered = normalized_query.lower()
        ranked = []
        for doc in self._documents:
            text = doc.get("text", "")
            score = self._lexical_score(lowered, text)
            if score:
                ranked.append((score, doc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [{"text": doc.get("text", ""), "metadata": doc} for _, doc in ranked[:k]]

    def handle(self, request: str, **kwargs: Any) -> AgentResponse:
        hits = self.search(request, k=int(kwargs.get("k", 3)))
        if not hits:
            content = "검색 결과를 찾지 못했습니다."
        else:
            snippets = []
            remaining = int(kwargs.get("max_context_chars", self.max_context_chars))
            for item in hits:
                text = re.sub(r"\s+", " ", item.get("text", "")).strip()
                if not text or remaining <= 0:
                    continue
                snippets.append(text[:remaining])
                remaining -= len(snippets[-1])
            content = "\n".join(snippets)
        return AgentResponse(agent=AgentKind.RAG, content=content, metadata={"hits": hits})

    def build_prompt_context(self, request: str, k: int = 2, max_context_chars: Optional[int] = None) -> str:
        response = self.handle(request, k=k, max_context_chars=max_context_chars or self.max_context_chars)
        content = response.content.strip()
        if not content or content == "검색 결과를 찾지 못했습니다.":
            return ""
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
        return f"[retrieved_context:{digest}]\n{content}"


class OrchestratorAgent(BaseAgent):
    def __init__(self, rag_agent: Optional[RAGAgent] = None, tool_agent: Optional[ToolAgent] = None) -> None:
        super().__init__("orchestrator")
        self.rag_agent = rag_agent or RAGAgent()
        self.tool_agent = tool_agent or ToolAgent()

    def classify_intent(self, request: str) -> Handoff:
        text = request.strip()
        lowered = text.lower()

        rag_keywords = ["문서", "지식", "검색", "찾아", "요약", "근거", "출처", "논문", "매뉴얼"]
        tool_keywords = ["블루투스", "알람", "타이머", "켜줘", "꺼줘", "메시지", "문자", "날씨", "기기", "제어", "배터리", "모터", "걸어", "앉아", "일어나", "정지"]

        if any(keyword in text for keyword in tool_keywords) or any(keyword in lowered for keyword in tool_keywords):
            return Handoff(target=AgentKind.TOOL, reason="device_or_action_request", payload={"request": text})

        if any(keyword in text for keyword in rag_keywords) or re.search(r"\b(rag|retrieve|vector|knowledge)\b", lowered):
            return Handoff(target=AgentKind.RAG, reason="knowledge_search_request", payload={"request": text})

        return Handoff(target=AgentKind.ORCHESTRATOR, reason="direct_answer", payload={"request": text})

    def handle(self, request: str, **kwargs: Any) -> AgentResponse:
        handoff = self.classify_intent(request)
        if handoff.target == AgentKind.RAG:
            return self.rag_agent.handle(request, **kwargs)
        if handoff.target == AgentKind.TOOL:
            tool_name = kwargs.get("tool_name")
            if not tool_name:
                if "배터리" in request:
                    tool_name = "check_robot_battery"
                elif any(keyword in request for keyword in ("모터", "걸어", "앉아", "일어나", "정지")):
                    tool_name = "control_quadruped_motor"
                else:
                    tool_name = "set_alarm" if "알람" in request else "search_web" if "검색" in request else "control_iot_device"
            return self.tool_agent.handle(request, tool_name=tool_name, arguments=kwargs.get("arguments", {}))
        return AgentResponse(agent=AgentKind.ORCHESTRATOR, content="direct route", metadata={"handoff": handoff.reason})


def build_orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent()
