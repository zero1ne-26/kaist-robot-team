from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan B GGUF inference with llama-cpp-python on Jetson")
    parser.add_argument("--model", required=True, help="Path to a quantized GGUF model, for example model-q4_k_m.gguf")
    parser.add_argument("--prompt", default="안녕, 오늘 날씨 이야기해줘.", help="Prompt text")
    parser.add_argument("--ctx-size", type=int, default=1024, help="Context length")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU. -1 uses all possible layers")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p")
    return parser.parse_args()


def main() -> None:
    from llama_cpp import Llama

    args = parse_args()
    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        n_threads=args.threads,
        verbose=False,
    )
    stream = llm.create_completion(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["</s>", "<|im_end|>"],
        stream=True,
    )
    for chunk in stream:
        text = chunk.get("choices", [{}])[0].get("text", "")
        if text:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()