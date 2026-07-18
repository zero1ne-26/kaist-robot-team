from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from interactive_scenario import (
    EXAONE_HRI_SYSTEM_PROMPT,
    INITIAL_SAD_QUESTION,
    call_exaone_contextual_response,
)
from loop4 import sanitize_tts_text, warm_up_models


DEFAULT_OUTPUT_DIR = Path("eval_pipeline/scenario_batch")

SCENARIOS: list[dict[str, str]] = [
    {"id": "sad_exam", "emotion": "sad", "stt": "오늘 시험을 망쳤어!!", "vision": "The user looks sad and gloomy after a bad exam."},
    {"id": "sad_friend", "emotion": "sad", "stt": "친구랑 크게 다퉜어", "vision": "The user looks sad and emotionally hurt."},
    {"id": "sad_rejection", "emotion": "sad", "stt": "면접에서 떨어졌어", "vision": "The user looks sad and disappointed."},
    {"id": "sad_tired", "emotion": "sad", "stt": "요즘 너무 지치고 아무것도 하기 싫어", "vision": "The user looks gloomy and exhausted."},
    {"id": "happy_project", "emotion": "happy", "stt": "오늘 드디어 프로젝트를 끝냈어!", "vision": "The user looks happy and proud."},
    {"id": "happy_prize", "emotion": "happy", "stt": "나 대회에서 상 받았어", "vision": "The user is smiling brightly."},
    {"id": "happy_friend", "emotion": "happy", "stt": "오랜만에 친구를 만나서 너무 좋아", "vision": "The user looks cheerful and relaxed."},
    {"id": "happy_success", "emotion": "happy", "stt": "코드가 드디어 돌아갔어", "vision": "The user looks relieved and happy."},
    {"id": "angry_bug", "emotion": "angry", "stt": "계속 에러가 나서 너무 화나", "vision": "The user looks angry and frustrated."},
    {"id": "angry_noise", "emotion": "angry", "stt": "옆방이 너무 시끄러워서 짜증나", "vision": "The user looks irritated."},
    {"id": "angry_delay", "emotion": "angry", "stt": "택배가 또 늦어서 화가 나", "vision": "The user looks angry and impatient."},
    {"id": "angry_team", "emotion": "angry", "stt": "팀원이 약속을 안 지켰어", "vision": "The user looks upset and angry."},
    {"id": "surprised_news", "emotion": "surprised", "stt": "갑자기 일정이 전부 바뀌었어", "vision": "The user looks surprised and startled."},
    {"id": "surprised_gift", "emotion": "surprised", "stt": "생각지도 못한 선물을 받았어", "vision": "The user looks surprised in a positive way."},
    {"id": "surprised_call", "emotion": "surprised", "stt": "방금 모르는 번호로 이상한 전화가 왔어", "vision": "The user looks surprised and slightly worried."},
    {"id": "surprised_alarm", "emotion": "surprised", "stt": "갑자기 알람이 크게 울렸어", "vision": "The user looks shocked by a sudden sound."},
    {"id": "neutral_schedule", "emotion": "neutral", "stt": "오늘 일정 알려줘", "vision": "The user looks neutral and calm."},
    {"id": "neutral_weather", "emotion": "neutral", "stt": "지금 날씨 어때", "vision": "The user has a neutral expression."},
    {"id": "neutral_music", "emotion": "neutral", "stt": "잔잔한 음악 틀어줘", "vision": "The user looks calm and neutral."},
    {"id": "neutral_question", "emotion": "neutral", "stt": "잠깐 이야기 좀 할래", "vision": "The user looks neutral but attentive."},
]

EMPATHY_WORDS = ("속상", "힘들", "안타깝", "괜찮", "옆", "도울", "위로", "천천히")
ENCOURAGEMENT_WORDS = ("힘내", "잘하실", "다음", "괜찮", "도울", "함께", "할 수", "응원")
CELEBRATION_WORDS = ("축하", "뿌듯", "좋은 일", "기쁘", "공유", "고생 많", "멋지")
CALMING_WORDS = ("차분", "화가", "짜증", "잠깐", "정리", "도와", "함께")
SURPRISE_WORDS = ("놀라", "갑자기", "괜찮", "무슨 일", "자세히", "확인")
NEUTRAL_WORDS = ("안녕하세요", "좋아요", "도와", "필요", "말씀", "함께")
FORBIDDEN_GENERIC = ("as an ai", "language model", "분석", "라벨", "markdown")

SITUATION_KEYWORDS = {
    "시험": ("시험", "망쳤"),
    "친구": ("친구", "다퉜"),
    "면접": ("면접", "떨어"),
    "지침": ("지치", "싫어"),
    "프로젝트": ("프로젝트", "끝냈"),
    "대회": ("대회", "상"),
    "코드": ("코드", "돌아"),
    "에러": ("에러", "화"),
    "택배": ("택배", "늦"),
    "팀원": ("팀원", "약속"),
    "일정": ("일정", "바뀌", "알려"),
    "선물": ("선물",),
    "전화": ("전화", "번호"),
    "알람": ("알람",),
    "날씨": ("날씨",),
    "음악": ("음악",),
    "이야기": ("이야기",),
}


def build_history_for_scenario(scenario: dict[str, str]) -> list[dict[str, str]]:
    initial_question = INITIAL_SAD_QUESTION if scenario["emotion"] == "sad" else "지금 기분이 어떠세요?"
    return [
        {"role": "system", "content": EXAONE_HRI_SYSTEM_PROMPT},
        {"role": "assistant", "content": initial_question},
        {"role": "user", "content": scenario["stt"]},
    ]


def contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word.lower() in text.lower() for word in words)


def situation_acknowledged(stt: str, response: str) -> bool:
    for keywords in SITUATION_KEYWORDS.values():
        if any(keyword in stt for keyword in keywords):
            return any(keyword in response for keyword in keywords)
    nouns = [token for token in re.findall(r"[가-힣]{2,}", stt) if token not in {"오늘", "너무", "드디어", "갑자기"}]
    return bool(nouns) and any(noun in response for noun in nouns[:2])


def forward_encouragement_ok(emotion: str, response: str) -> bool:
    if emotion == "happy":
        return contains_any(response, CELEBRATION_WORDS) and "무슨 일 있으세요" not in response
    if emotion == "sad":
        return contains_any(response, EMPATHY_WORDS) and contains_any(response, ENCOURAGEMENT_WORDS)
    if emotion == "angry":
        return contains_any(response, CALMING_WORDS) and contains_any(response, ENCOURAGEMENT_WORDS)
    if emotion == "surprised":
        return contains_any(response, SURPRISE_WORDS)
    if emotion == "neutral":
        return contains_any(response, NEUTRAL_WORDS) and "무슨 일 있으세요" not in response
    return False


def tts_safe(response: str) -> bool:
    return sanitize_tts_text(response) == response.strip()


def evaluate_response(scenario: dict[str, str], response: str) -> dict[str, Any]:
    empathy_ok = situation_acknowledged(scenario["stt"], response)
    encouragement_ok = forward_encouragement_ok(scenario["emotion"], response)
    tts_ok = tts_safe(response)
    generic_break = contains_any(response, FORBIDDEN_GENERIC)
    score = 0
    score += 35 if empathy_ok else 0
    score += 35 if encouragement_ok else 0
    score += 20 if tts_ok else 0
    score += 10 if not generic_break else 0
    issues: list[str] = []
    if not empathy_ok:
        issues.append("specific STT situation not acknowledged")
    if not encouragement_ok:
        issues.append("missing emotion-specific encouragement")
    if not tts_ok:
        issues.append("unsafe TTS characters remain")
    if generic_break:
        issues.append("generic AI/meta wording detected")
    return {
        "empathy_ok": empathy_ok,
        "encouragement_ok": encouragement_ok,
        "tts_ok": tts_ok,
        "generic_break": generic_break,
        "score": score,
        "issues": "; ".join(issues) or "ok",
    }


def run_one_scenario(scenario: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    history = build_history_for_scenario(scenario)
    response = call_exaone_contextual_response(history, scenario["vision"], emotion=scenario["emotion"])
    evaluation = evaluate_response(scenario, response)
    return {
        "id": scenario["id"],
        "emotion": scenario["emotion"],
        "stt": scenario["stt"],
        "vision": scenario["vision"],
        "response": response,
        "latency_sec": round(time.perf_counter() - started, 3),
        **evaluation,
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scenario_batch_results.csv"
    report_path = output_dir / "scenario_batch_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_score = mean(row["score"] for row in rows)
    avg_latency = mean(row["latency_sec"] for row in rows)
    pass_rate = sum(1 for row in rows if row["score"] >= 80) / len(rows) * 100
    lines = [
        "# Scenario Batch Evaluation",
        "",
        "## Optimized System Prompt",
        "",
        "```text",
        EXAONE_HRI_SYSTEM_PROMPT.strip(),
        "```",
        "",
        "## Scorecard",
        f"- Samples: {len(rows)}",
        f"- Average score: {avg_score:.1f} / 100",
        f"- Pass rate score>=80: {pass_rate:.1f}%",
        f"- Average EXAONE latency: {avg_latency:.2f}s",
        "",
        "## Matrix",
        "",
        "| id | emotion | score | empathy | encouragement | tts_safe | latency_s | issue | response |",
        "|---|---|---:|---|---|---|---:|---|---|",
    ]
    for row in rows:
        safe_response = row["response"].replace("|", " ")
        lines.append(
            f"| {row['id']} | {row['emotion']} | {row['score']} | {row['empathy_ok']} | "
            f"{row['encouragement_ok']} | {row['tts_ok']} | {row['latency_sec']:.2f} | {row['issues']} | {safe_response} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="20-sample multi-turn HRI scenario evaluator for EXAONE prompt quality")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV and Markdown reports")
    parser.add_argument("--no-warmup", action="store_true", help="Skip Ollama keep-alive warm-up")
    args = parser.parse_args()

    if not args.no_warmup:
        print("[WARMUP] keeping Ollama models loaded...")
        warm_up_models()

    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"[{index:02d}/{len(SCENARIOS)}] {scenario['emotion']} | {scenario['stt']}")
        row = run_one_scenario(scenario)
        rows.append(row)
        print(f"    score={row['score']} latency={row['latency_sec']:.2f}s response={row['response']}")

    report_path = write_outputs(rows, Path(args.output_dir))
    print("\n===== Scenario Batch Scorecard =====")
    print(f"samples={len(rows)}")
    print(f"average_score={mean(row['score'] for row in rows):.1f}/100")
    print(f"average_latency={mean(row['latency_sec'] for row in rows):.2f}s")
    print(f"pass_rate_score>=80={sum(1 for row in rows if row['score'] >= 80) / len(rows) * 100:.1f}%")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()