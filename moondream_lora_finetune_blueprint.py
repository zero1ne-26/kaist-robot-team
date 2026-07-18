from __future__ import annotations

import argparse
import json
from pathlib import Path


"""
Blueprint for LoRA/QLoRA-style facial-emotion tuning.

Moondream's exact Hugging Face class can vary by release. Treat this file as a
structured starting point: replace MODEL_ID and processor/model loading with the
specific Moondream HF checkpoint API you choose, then keep the dataset contract.

Dataset JSONL example:
{"image": "data/sad_001.jpg", "label": "Sad"}
{"image": "data/anxious_001.jpg", "label": "Anxious"}
"""


MODEL_ID = "vikhyatk/moondream2"
PROMPT = "Output exactly one emotion keyword: Happy, Sad, Angry, Anxious, Surprised, Neutral."


def load_records(jsonl_path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def train_lora(jsonl_path: str, output_dir: str) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    records = load_records(jsonl_path)
    dataset = Dataset.from_list(records)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    def formatting_func(example: dict[str, str]) -> str:
        _ = Image.open(example["image"]).convert("RGB")
        return f"User: <image>\n{PROMPT}\nAssistant: {example['label']}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        tokenizer=getattr(processor, "tokenizer", None),
        max_seq_length=256,
        packing=False,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Moondream facial-emotion LoRA fine-tuning blueprint")
    parser.add_argument("--jsonl", required=True, help="JSONL with image and label fields")
    parser.add_argument("--output-dir", default="./moondream_emotion_lora")
    args = parser.parse_args()
    train_lora(args.jsonl, args.output_dir)


if __name__ == "__main__":
    main()