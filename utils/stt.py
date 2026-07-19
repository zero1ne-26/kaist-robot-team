from __future__ import annotations


def listen_from_terminal(prompt: str = "[STT 입력 시뮬레이션] 사용자 말: ", default_text: str = "") -> str:
    """Return terminal-entered text for demos that simulate an STT result."""
    user_text = input(prompt).strip()
    return user_text or default_text
