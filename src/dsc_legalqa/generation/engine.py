"""Qwen3.5-2B + P70 LoRA generation engine."""

from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dsc_legalqa.data.schema import LegalAnswer, PackedEvidence
from dsc_legalqa.generation.prompt import build_generation_prompt
from dsc_legalqa.postprocess.duplicate_guard import sanitize_duplicate_loops
from dsc_legalqa.utils.config import sha256_file

MODEL_ID = "Qwen/Qwen3.5-2B"
PINNED_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPECTED_ADAPTER_SHA = "193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c"
EXPECTED_CONFIG_SHA = "2761928c3f17aa894bd2e7793db0d00a2e7c1b0821a7d864dc7bf411dc603f37"

DECODE_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 1536,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "use_cache": True,
}


class GenerationEngine:
    """Production generator engine with greedy decoding and repetition guard."""

    def __init__(
        self,
        base_model_id: str = MODEL_ID,
        revision: str = PINNED_REVISION,
        adapter_dir: Path | str = "artifacts/p70_adapter",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.base_model_id = base_model_id
        self.revision = revision
        self.adapter_dir = Path(adapter_dir)
        self.device = device
        self.model = None
        self.tokenizer = None

    def verify_adapter_hashes(self):
        w_path = self.adapter_dir / "adapter_model.safetensors"
        c_path = self.adapter_dir / "adapter_config.json"
        w_sha = sha256_file(w_path)
        c_sha = sha256_file(c_path)
        if w_sha != EXPECTED_ADAPTER_SHA:
            raise RuntimeError(f"Adapter weights SHA mismatch: {w_sha} != {EXPECTED_ADAPTER_SHA}")
        if c_sha != EXPECTED_CONFIG_SHA:
            raise RuntimeError(f"Adapter config SHA mismatch: {c_sha} != {EXPECTED_CONFIG_SHA}")

    def load_model(self):
        if self.model is None:
            self.verify_adapter_hashes()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                revision=self.revision,
                use_fast=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                revision=self.revision,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                device_map=self.device,
            )
            self.model = PeftModel.from_pretrained(
                base,
                str(self.adapter_dir),
            )
            self.model.eval()

    def generate_answer(
        self,
        qid: str,
        question: str,
        packed_evidence: PackedEvidence,
    ) -> LegalAnswer:
        self.load_model()
        prompt = build_generation_prompt(question, packed_evidence.formatted_context)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                **DECODE_CONFIG,
            )

        continuation_ids = output_ids[0][input_len:].cpu().tolist()
        # Apply production duplicate guard
        cleaned_ids, guard_applied = sanitize_duplicate_loops(continuation_ids)
        answer_text = self.tokenizer.decode(cleaned_ids, skip_special_tokens=True).strip()

        return LegalAnswer(
            qid=qid,
            question=question,
            answer=answer_text,
            evidence_ids=packed_evidence.chunk_ids,
            tokens_generated=len(cleaned_ids),
            duplicate_guard_applied=guard_applied,
        )
