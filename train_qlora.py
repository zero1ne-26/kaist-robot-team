from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainingConfig:
    model_name: str = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    dataset_path: str = os.getenv("DATASET_PATH", "jetson_dataset.jsonl")
    output_dir: str = os.getenv("OUTPUT_DIR", "./saved_lora_model")
    max_steps: int = int(os.getenv("MAX_STEPS", "0"))
    num_train_epochs: float = float(os.getenv("NUM_TRAIN_EPOCHS", "3"))
    per_device_train_batch_size: int = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
    target_effective_batch_size: int = int(os.getenv("TARGET_EFFECTIVE_BATCH_SIZE", "8"))
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "0"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    lora_rank: int = int(os.getenv("LORA_R", "8"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "16"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.08"))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.05"))
    optim: str = os.getenv("TRAIN_OPTIM", "paged_adamw_8bit")
    gradient_checkpointing: bool = os.getenv("GRADIENT_CHECKPOINTING", "1").strip().lower() in {"1", "true", "yes", "on"}
    gradient_checkpointing_use_reentrant: bool = os.getenv("GRADIENT_CHECKPOINTING_USE_REENTRANT", "0").strip().lower() in {"1", "true", "yes", "on"}
    max_memory_gib: float = float(os.getenv("MAX_MEMORY_GIB", "5.5"))


def load_jsonl_dataset(path: str) -> Dataset:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example: Dict[str, Any]) -> str:
    rendered_messages = []
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        rendered_messages.append(f"{role}: {content}")
    return "\n".join(rendered_messages)


def build_model_and_tokenizer(model_name: str, max_memory_gib: float, gradient_checkpointing: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_memory = None
    if max_memory_gib > 0 and torch.cuda.is_available():
        max_memory = {0: f"{max_memory_gib:.1f}GiB", "cpu": "24GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    return model, tokenizer


def build_lora_model(model, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora_config)


def resolve_gradient_accumulation_steps(
    per_device_batch_size: int,
    target_effective_batch_size: int,
    explicit_steps: int,
) -> int:
    if explicit_steps > 0:
        return explicit_steps
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    return max(1, math.ceil(target_effective_batch_size / per_device_batch_size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson QLoRA fine-tuning script")
    parser.add_argument("--model-name", default=TrainingConfig.model_name, help="Base model name or path")
    parser.add_argument("--dataset-path", default=TrainingConfig.dataset_path, help="JSONL dataset path")
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir, help="LoRA adapter output directory")
    parser.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps, help="Optional training step cap. 0 uses epochs")
    parser.add_argument("--epochs", type=float, default=TrainingConfig.num_train_epochs, help="Epochs for full fine-tuning")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.per_device_train_batch_size, help="Per-device batch size")
    parser.add_argument("--target-effective-batch-size", type=int, default=TrainingConfig.target_effective_batch_size, help="Effective batch size target used when --grad-accum is 0")
    parser.add_argument("--grad-accum", type=int, default=TrainingConfig.gradient_accumulation_steps, help="Gradient accumulation steps. 0 auto-computes from target effective batch size")
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=TrainingConfig.max_seq_length, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=TrainingConfig.lora_rank, help="LoRA rank. Jetson-friendly default is 8")
    parser.add_argument("--lora-alpha", type=int, default=TrainingConfig.lora_alpha, help="LoRA alpha. Default keeps alpha/r=2")
    parser.add_argument("--lora-dropout", type=float, default=TrainingConfig.lora_dropout, help="LoRA dropout for overfitting control")
    parser.add_argument("--warmup-ratio", type=float, default=TrainingConfig.warmup_ratio, help="Warmup ratio before cosine decay")
    parser.add_argument("--optim", default=TrainingConfig.optim, help="Optimizer. Jetson OOM-safe default: paged_adamw_8bit")
    parser.add_argument("--max-memory-gib", type=float, default=TrainingConfig.max_memory_gib, help="CUDA memory cap for device_map auto. 0 disables cap")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--gradient-checkpointing-use-reentrant", action="store_true", default=TrainingConfig.gradient_checkpointing_use_reentrant, help="Use reentrant checkpointing. Default false reduces k-bit training issues")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional dataset cap for quick dry-runs. 0 uses all samples")
    parser.add_argument("--dry-run", action="store_true", help="Force a minimal one-step run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_steps=1 if args.dry_run else args.max_steps,
        num_train_epochs=1 if args.dry_run else args.epochs,
        per_device_train_batch_size=args.batch_size,
        target_effective_batch_size=args.target_effective_batch_size,
        gradient_accumulation_steps=resolve_gradient_accumulation_steps(
            args.batch_size,
            args.target_effective_batch_size,
            args.grad_accum,
        ),
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        max_memory_gib=args.max_memory_gib,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        gradient_checkpointing_use_reentrant=args.gradient_checkpointing_use_reentrant,
    )

    from transformers import TrainingArguments
    from trl import SFTTrainer

    dataset = load_jsonl_dataset(config.dataset_path)
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    model, tokenizer = build_model_and_tokenizer(
        config.model_name,
        max_memory_gib=config.max_memory_gib,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    model = build_lora_model(model, config.lora_rank, config.lora_alpha, config.lora_dropout)
    model.print_trainable_parameters()

    use_max_steps = config.max_steps > 0

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="epoch" if not use_max_steps else "steps",
        save_steps=max(1, config.max_steps) if use_max_steps else 500,
        save_total_limit=1,
        max_steps=config.max_steps if use_max_steps else -1,
        num_train_epochs=config.num_train_epochs,
        optim=config.optim,
        lr_scheduler_type="cosine",
        bf16=False,
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": config.gradient_checkpointing_use_reentrant},
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        group_by_length=True,
        warmup_ratio=config.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=format_example,
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Saved LoRA adapter to {config.output_dir}")


if __name__ == "__main__":
    main()
