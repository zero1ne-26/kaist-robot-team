from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multimodal_robot_loop import DEFAULT_CONSECUTIVE_FRAMES, DEFAULT_TRIGGER_THRESHOLD, run_robot_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular MediaPipe -> LLM -> TTS robot loop")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD)
    parser.add_argument("--consecutive-frames", type=int, default=DEFAULT_CONSECUTIVE_FRAMES)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run until q/camera stop")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_robot_loop(
        camera_index=args.camera_index,
        threshold=args.threshold,
        consecutive_frames=args.consecutive_frames,
        play_audio=not args.no_play,
        warmup=not args.no_warmup,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
