from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_core import run_pipeline
from stt_tts_pipeline import (
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_FIRST_CHUNK_MIN_CHARS,
    DEFAULT_MIC_PHRASE_TIME_LIMIT,
    DEFAULT_MIC_TIMEOUT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PLAYBACK_PREBUFFER_CHUNKS,
    DEFAULT_QUEUE_MAXSIZE,
    DEFAULT_RAG_CONTEXT_CHARS,
    StreamingVoicePipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular STT -> LLM -> TTS pipeline")
    parser.add_argument("-o", "--output", default="response.mp3", help="저장할 TTS 오디오 파일 경로")
    parser.add_argument("--input-type", choices=["mic", "file"], default="mic", help="STT 입력 타입 선택")
    parser.add_argument("--file-path", default=None, help="input-type=file일 때 사용할 오디오 파일 경로")
    parser.add_argument("--mic-timeout", type=float, default=DEFAULT_MIC_TIMEOUT)
    parser.add_argument("--mic-phrase-time-limit", type=float, default=DEFAULT_MIC_PHRASE_TIME_LIMIT)
    parser.add_argument("--streaming", action="store_true", help="저지연 async LLM/TTS 스트리밍 사용")
    parser.add_argument("--queue-maxsize", type=int, default=DEFAULT_QUEUE_MAXSIZE)
    parser.add_argument("--first-chunk-min-chars", type=int, default=DEFAULT_FIRST_CHUNK_MIN_CHARS)
    parser.add_argument("--chunk-min-chars", type=int, default=DEFAULT_CHUNK_MIN_CHARS)
    parser.add_argument("--chunk-max-chars", type=int, default=DEFAULT_CHUNK_MAX_CHARS)
    parser.add_argument("--playback-prebuffer-chunks", type=int, default=DEFAULT_PLAYBACK_PREBUFFER_CHUNKS)
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--rag-corpus-path", default="./knowledge_base")
    parser.add_argument("--rag-context-chars", type=int, default=DEFAULT_RAG_CONTEXT_CHARS)
    parser.add_argument("--enable-barge-in", action="store_true")
    parser.add_argument("--max-turns", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
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
        return

    run_pipeline(
        output_path=args.output,
        model=args.model,
        input_type=args.input_type,
        file_path=args.file_path,
        mic_timeout=args.mic_timeout,
        mic_phrase_time_limit=args.mic_phrase_time_limit,
    )


if __name__ == "__main__":
    main()
