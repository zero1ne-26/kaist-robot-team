from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model for Jetson inference")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name or local path")
    parser.add_argument("--adapter-path", default="./saved_lora_model", help="LoRA adapter directory")
    parser.add_argument("--output-dir", default="./merged_lora_model", help="Merged model output directory")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value")
    return parser.parse_args()


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_dir)

    print(f"Merged LoRA adapter saved to {output_dir}")
    print("For 4-bit/8-bit inference, load this directory with serve_vllm_inference.py --quantization bitsandbytes.")


if __name__ == "__main__":
    main()