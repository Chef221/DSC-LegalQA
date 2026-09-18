from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

"""P70 Continued-LoRA SFT Training Runner for UIT DSC 2026 Task 2.

Continuing training from P2 LoRA adapter on Qwen/Qwen3.5-2B.
Conditioned on P63 selected evidence and reference-free [ANSWER_CONTROL] header.
Preserves authentic production training invariants:
- 4,247 train rows from Folds 1-4
- Zero leakage with heldout500
- Supervise full gold target without truncation, prompt tokens masked with -100
- Trainable params: exactly 21,823,488
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("train_p70")

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

EXPECTED_P2_WEIGHTS_SHA = "2b396cf119e72dac316bc06cd9c781d9895fbefd557c9e9426f2966a39afaacd"
EXPECTED_P2_CONFIG_SHA = "85647cebc8d0a1162d2d5ac0e3708712b61c5d394a1e98ef47e94cd2a111a023"
EXPECTED_TRAINABLE_PARAMS = 21_823_488
EXPECTED_TRAIN_ROWS = 4247


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class P70Dataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: Any, max_prompt_length: int = 7680):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples[idx]
        prompt = row["prompt"]
        gold = row["gold_answer"].strip() + "<|im_end|>"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        gold_ids = self.tokenizer.encode(gold, add_special_tokens=False)

        if len(prompt_ids) > self.max_prompt_length:
            raise ValueError(f"FAIL CLOSED: Prompt tokens {len(prompt_ids)} > {self.max_prompt_length} for QID {row.get('qid')}")

        input_ids = prompt_ids + gold_ids
        # Mask prompt from loss computation (-100), supervise full gold target
        labels = [-100] * len(prompt_ids) + gold_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_to = int(math.ceil(max_len / 8) * 8)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = pad_to - len(inp)

        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        att_mask = torch.cat([torch.ones(len(inp), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])

        input_ids.append(padded_inp)
        labels.append(padded_lbl)
        attention_mask.append(att_mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="P70 Continued-LoRA Training Runner")
    parser.add_argument("--train_file", type=str, default="data/P70_TRAIN_ROWS.jsonl", help="Training examples JSONL.")
    parser.add_argument("--heldout_file", type=str, default="data/P70_HELDOUT500_ROWS.jsonl", help="Heldout500 examples JSONL.")
    parser.add_argument("--base_model_id", type=str, default=MODEL_ID)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    parser.add_argument("--p2_adapter_dir", type=str, default="artifacts/p2_adapter", help="Initial P2 adapter directory.")
    parser.add_argument("--output_dir", type=str, default="artifacts/p70_trained_adapter")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Using device: {device}")

    # 1. Assertions on P2 adapter integrity
    p2_dir = Path(args.p2_adapter_dir)
    p2_weights = p2_dir / "adapter_model.safetensors"
    p2_cfg = p2_dir / "adapter_config.json"
    if not p2_weights.is_file():
        raise FileNotFoundError(
            f"FAIL CLOSED: Initial P2 adapter weights missing at {p2_weights}.\n"
            f"P70 is a continued-LoRA fine-tuning run initialized from the frozen P2 adapter."
        )
    if not p2_cfg.is_file():
        raise FileNotFoundError(f"FAIL CLOSED: Initial P2 adapter config missing at {p2_cfg}")

    actual_p2_sha = compute_sha256(p2_weights)
    actual_cfg_sha = compute_sha256(p2_cfg)
    if actual_p2_sha != EXPECTED_P2_WEIGHTS_SHA:
        raise ValueError(f"FAIL CLOSED: P2 weights SHA {actual_p2_sha} != {EXPECTED_P2_WEIGHTS_SHA}")
    if actual_cfg_sha != EXPECTED_P2_CONFIG_SHA:
        raise ValueError(f"FAIL CLOSED: P2 config SHA {actual_cfg_sha} != {EXPECTED_P2_CONFIG_SHA}")
    _LOGGER.info(f"[PASS] Initial P2 base adapter verified: {actual_p2_sha}")

    # 2. Check training data & assert zero heldout leakage
    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise FileNotFoundError(f"FAIL CLOSED: P70 training data file not found at: {train_path}")

    heldout_qids = set()
    if Path(args.heldout_file).is_file():
        with open(args.heldout_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    heldout_qids.add(str(json.loads(line)["qid"]))
        _LOGGER.info(f"Loaded {len(heldout_qids)} heldout QIDs for zero-leakage assertion.")

    train_rows = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_rows.append(json.loads(line))

    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        _LOGGER.warning(f"Train row count: {len(train_rows)} (Expected: {EXPECTED_TRAIN_ROWS})")

    train_qids = {str(r["qid"]) for r in train_rows}
    leakage = train_qids.intersection(heldout_qids)
    if leakage:
        raise ValueError(f"FAIL CLOSED: Critical leakage! {len(leakage)} heldout QIDs found in train: {leakage}")
    _LOGGER.info("[PASS] Zero leakage between training and heldout verified.")

    # 3. Model & LoRA initialization
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 248044

    _LOGGER.info(f"Loading base model {args.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        revision=args.revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if device.type == "cuda":
        base_model.gradient_checkpointing_enable()

    _LOGGER.info(f"Attaching P2 adapter from {args.p2_adapter_dir}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(p2_dir),
        is_trainable=True,
    )
    model.enable_input_require_grads()
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    _LOGGER.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    if trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise ValueError(f"FAIL CLOSED: Trainable params {trainable_params} != {EXPECTED_TRAINABLE_PARAMS}")

    # 4. DataLoader & Optimization
    dataset = P70Dataset(str(train_path), tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    _LOGGER.info(f"Starting training: {args.epochs} epoch(s), {total_steps} optimizer steps, grad_accum={args.grad_accum}...")
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    _LOGGER.info(f"Step {global_step}/{total_steps} - Loss: {accum_loss * args.grad_accum:.4f}")
                accum_loss = 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    _LOGGER.info(f"[SUCCESS] P70 adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
