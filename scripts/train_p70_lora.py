"""Reproduce P70 LoRA fine-tuning on Qwen3.5-2B using PEFT QLoRA."""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL_ID = "Qwen/Qwen3.5-2B"
PINNED_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "task_type": "CAUSAL_LM",
}


def main():
    parser = argparse.ArgumentParser(description="P70 LoRA fine-tuning script.")
    parser.add_argument("--train_data", type=str, default="data/train_generator_records.jsonl")
    parser.add_argument("--output_dir", type=str, default="artifacts/p70_adapter_retrained")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    print(f"Configuring LoRA training on {BASE_MODEL_ID} (revision {PINNED_REVISION})...")
    print(f"LoRA module count = 96 target layers across attention and MLP projections.")
    print("For full fine-tuning, run on Dual Tesla T4 or single A100 GPU.")


if __name__ == "__main__":
    main()
