from __future__ import annotations

import argparse
import importlib
import os
from typing import Iterable


def configure_jetson_vllm_environment() -> None:
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    os.environ.setdefault("VLLM_USE_FLASHINFER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency vLLM inference for a merged Jetson LLM")
    parser.add_argument("--model", default="./merged_lora_model", help="Merged model path or HF model id")
    parser.add_argument("--prompt", default="안녕, 오늘 날씨 이야기해줘.", help="Prompt to generate from")
    parser.add_argument("--quantization", default="bitsandbytes", choices=["bitsandbytes", "none"], help="vLLM quantization backend")
    parser.add_argument("--load-format", default="bitsandbytes", choices=["bitsandbytes", "auto"], help="vLLM load format")
    parser.add_argument("--max-model-len", type=int, default=1024, help="Context length cap for Jetson VRAM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82, help="vLLM GPU memory fraction")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p")
    return parser.parse_args()


def _iter_output_texts(outputs: Iterable[object]) -> Iterable[str]:
    for output in outputs:
        candidates = getattr(output, "outputs", []) or []
        for candidate in candidates:
            text = getattr(candidate, "text", "")
            if text:
                yield text


def main() -> None:
    configure_jetson_vllm_environment()
    try:
        vllm_module = importlib.import_module("vllm")
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this Python environment. Install a Jetson-compatible vLLM build, "
            "or use infer_llama_cpp.py with a quantized GGUF model."
        ) from exc
    LLM = vllm_module.LLM
    SamplingParams = vllm_module.SamplingParams

    args = parse_args()
    quantization = None if args.quantization == "none" else args.quantization
    load_format = "auto" if quantization is None else args.load_format

    try:
        llm = LLM(
            model=args.model,
            quantization=quantization,
            load_format=load_format,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "vLLM initialization failed. On Jetson ARM64, check CUDA/PyTorch/vLLM wheel compatibility. "
            "Fallback path: convert the merged model to GGUF and run infer_llama_cpp.py."
        ) from exc
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["<|im_end|>", "</s>"],
    )
    outputs = llm.generate([args.prompt], sampling_params)
    for text in _iter_output_texts(outputs):
        print(text.strip())


if __name__ == "__main__":
    main()