# Jetson QLoRA Fine-Tuning Report

## Goal

Build a small, Jetson-friendly behavior-cloning pipeline for the Genie assistant so the runtime can stay fast while the personality and tool-routing behavior can still be refined offline.

## What Was Added

- `generate_dataset.py` creates JSONL training data for three buckets:
  - direct general-knowledge answers
  - tool-triggering requests
  - unknowable or out-of-domain prompts that should be refused
- `train_qlora.py` provides the training entrypoint for QLoRA adapter fine-tuning.
- `audio_manager.py` can optionally load a LoRA adapter for local inference.

## Training Design

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Quantization: 4-bit NF4 with double quantization
- Attention: SDPA
- Optimizer: paged AdamW 8-bit
- Memory strategy: gradient checkpointing
- LoRA target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

## Runtime Integration

- The default runtime path still uses Ollama.
- LoRA loading is opt-in through environment flags.
- If the adapter is not present, the assistant continues to run normally without any training artifacts.

## Validation

- Dataset generation completed successfully with 150 examples in `jetson_dataset.jsonl`.
- The training script now prints `--help` without requiring training dependencies.
- The module imports cleanly even when the ML stack is absent.
- Runtime direct-question streaming still works and avoids tool overhead for common knowledge.

## Current Limitation

- The current workspace environment does not have the full fine-tuning stack installed, so an actual training run is not yet possible here.
- Required packages for a real run are `torch`, `transformers`, `datasets`, `peft`, and `trl`.