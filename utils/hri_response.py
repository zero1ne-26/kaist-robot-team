from __future__ import annotations

import requests

from utils.loop4 import KEEP_ALIVE, OLLAMA_CHAT_URL, REASONING_MODEL, REQUEST_TIMEOUT_SECONDS, sanitize_tts_text


EXAONE_HRI_SYSTEM_PROMPT = """너는 자비스라는 한국어 로봇 비서다.
너의 성격은 가까운 친구처럼 따뜻하고, 로봇 비서처럼 차분하며, 먼저 힘을 주는 정서 지능형 HRI 파트너다.

반드시 지킬 응답 구조:
1. 사용자의 말을 구체적으로 반영한다. 시험, 프로젝트, 싸움, 놀란 일처럼 핵심 사건을 직접 언급한다.
2. 감정을 인정한다. "정말 속상했겠어요", "그건 뿌듯하겠어요"처럼 마음을 먼저 받아준다.
3. 앞으로의 힘을 준다. 다음 행동, 회복, 축하 질문, 격려 중 하나를 반드시 덧붙인다.
4. 한국어 1~2문장만 말한다. 이모지, markdown, 분석 라벨, 영어 설명은 금지한다.
5. 사용자가 힘들어할 때는 절대 훈계하지 말고, "다음번엔 꼭 더 잘하실 수 있을 거예요", "제가 옆에서 도울게요"처럼 적극적으로 격려한다.

감정별 행동 규칙:
- sad: "무슨 일 있으세요?"를 반복하지 말고, 이미 말한 이유를 반영해 깊게 위로하고 회복을 격려한다.
- happy: 걱정 질문 금지. 함께 기뻐하고, 더 들려달라고 자연스럽게 묻는다.
- angry: 화난 이유를 인정하고, 편을 들어주되 차분해질 작은 다음 행동을 제안한다.
- surprised: 놀란 상황을 받아주고, 괜찮은지 확인하며 자세히 말해달라고 한다.
- neutral: 부담 없이 인사하고, 필요한 일이 있으면 돕겠다고 말한다.

좋은 예시:
사용자: 오늘 시험을 망쳤어!!
자비스: 시험을 망쳐서 정말 속상했겠어요. 그래도 오늘 하나로 끝난 건 아니니까, 다음번엔 꼭 더 잘하실 수 있을 거예요. 힘내세요!
사용자: 오늘 드디어 프로젝트를 끝냈어!
자비스: 드디어 프로젝트를 끝냈다니 정말 뿌듯하겠어요. 고생 많았어요, 어떤 부분이 제일 만족스러웠는지도 저랑 공유해요!
"""


def build_contextual_messages(
    conversation_history: list[dict[str, str]],
    vision_description: str,
    emotion: str = "sad",
) -> list[dict[str, str]]:
    """Combine system prompt, visual context, and multi-turn history for EXAONE."""
    system_message = next(
        (turn for turn in conversation_history if turn.get("role") == "system"),
        {"role": "system", "content": EXAONE_HRI_SYSTEM_PROMPT},
    )
    dialogue_turns = [turn for turn in conversation_history if turn.get("role") != "system"]
    messages = [
        system_message,
        {
            "role": "user",
            "content": (
                "현재 시각 맥락: "
                f"{vision_description}\n"
                f"감정 라벨: {emotion}\n"
                "이전 대화와 사용자의 방금 발화를 바탕으로 로봇이 지금 말할 문장만 생성해. "
                "응답에는 상황 반영, 감정 인정, 앞으로의 격려를 모두 포함해."
            ),
        },
    ]
    messages.extend(dialogue_turns)
    return messages


def call_exaone_contextual_response(
    conversation_history: list[dict[str, str]],
    vision_description: str,
    emotion: str = "sad",
) -> str:
    """Generate an empathetic Korean HRI response with EXAONE and keep_alive enabled."""
    payload = {
        "model": REASONING_MODEL,
        "messages": build_contextual_messages(conversation_history, vision_description, emotion=emotion),
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.25,
            "num_predict": 96,
            "num_ctx": 1024,
        },
    }
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError("EXAONE 응답 시간이 초과되었습니다.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"EXAONE 호출 실패: {exc}") from exc

    text = str(response.json().get("message", {}).get("content", "")).strip()
    if not text:
        raise RuntimeError("EXAONE이 빈 응답을 반환했습니다.")
    return sanitize_tts_text(text)
