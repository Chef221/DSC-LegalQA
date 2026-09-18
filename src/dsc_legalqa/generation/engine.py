"""Qwen3.5-2B + P70 LoRA generation engine matching accepted production runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch

try:
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
    HAS_QWEN35_COND = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_QWEN35_COND = False

from peft import PeftModel
from dsc_legalqa.data.schema import LegalAnswer, PackedEvidence
from dsc_legalqa.generation.prompt import build_generation_prompt
from dsc_legalqa.postprocess.duplicate_guard import sanitize_duplicate_loops
from dsc_legalqa.utils.config import sha256_file

_LOGGER = logging.getLogger(__name__)

# Frozen production model constants
MODEL_ID = "Qwen/Qwen3.5-2B"
PINNED_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPECTED_ADAPTER_SHA = "193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c"
EXPECTED_CONFIG_SHA = "2761928c3f17aa894bd2e7793db0d00a2e7c1b0821a7d864dc7bf411dc603f37"
EXPECTED_ADAPTER_TENSORS = 192

# Authoritative baseline greedy decoding configuration
BASELINE_DECODE = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 1536,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "use_cache": True,
}


def verify_exact_adapter_binding(model: Any, adapter_dir: Path | str, device_label: str = "GPU") -> dict[str, Any]:
    """Verifies that all 192 saved LoRA adapter tensors bind exactly to the live model."""
    from safetensors import safe_open

    adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
    saved = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            saved[k] = f.get_tensor(k)

    if len(saved) != EXPECTED_ADAPTER_TENSORS:
        raise RuntimeError(f"Saved adapter tensors != {EXPECTED_ADAPTER_TENSORS}: {len(saved)}")

    params = dict(model.named_parameters())
    missing = []
    mismatched = []
    checked = 0
    for src_key, saved_tensor in saved.items():
        dst_key = src_key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        if dst_key not in params:
            missing.append(dst_key)
            continue
        live = params[dst_key].detach().cpu().to(dtype=saved_tensor.dtype)
        if tuple(live.shape) != tuple(saved_tensor.shape) or not torch.equal(saved_tensor, live):
            mismatched.append(dst_key)
            continue
        checked += 1

    if checked != EXPECTED_ADAPTER_TENSORS or missing or mismatched:
        raise RuntimeError(
            f"Adapter exact binding failed {device_label}: checked={checked} missing={len(missing)} mismatched={len(mismatched)}"
        )

    _LOGGER.info(f"[PASS] Exact adapter tensor binding {device_label}: {checked}/{EXPECTED_ADAPTER_TENSORS}")
    return {"device": device_label, "checked": f"{checked}/{EXPECTED_ADAPTER_TENSORS}", "missing": len(missing), "mismatched": len(mismatched)}


def compute_raw_cap_hit(raw_tokens_count: int, max_new_tokens: int, eos_seen: bool) -> bool:
    """raw_cap_hit = (raw_tokens_count >= max_new_tokens) and (not eos_seen)."""
    return bool(raw_tokens_count >= max_new_tokens and not eos_seen)


class GenerationEngine:
    """Production generator engine with Qwen3.5-2B + P70 LoRA, greedy decoding, and repetition guard."""

    def __init__(
        self,
        base_model_id: str = MODEL_ID,
        revision: str = PINNED_REVISION,
        adapter_dir: Path | str = "artifacts/p70_adapter",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        decode_config: dict[str, Any] | None = None,
        allow_compatibility_fallback: bool = False,
    ):
        self.base_model_id = base_model_id
        self.revision = revision
        self.adapter_dir = Path(adapter_dir)
        self.device = device
        self.decode_config = decode_config or dict(BASELINE_DECODE)
        self.allow_compatibility_fallback = allow_compatibility_fallback
        self.model = None
        self.tokenizer = None

    def verify_adapter_hashes(self):
        w_path = self.adapter_dir / "adapter_model.safetensors"
        c_path = self.adapter_dir / "adapter_config.json"
        if not w_path.exists():
            raise FileNotFoundError(f"Adapter weights not found: {w_path}")
        if not c_path.exists():
            raise FileNotFoundError(f"Adapter config not found: {c_path}")

        w_sha = sha256_file(w_path)
        c_sha = sha256_file(c_path)
        if w_sha != EXPECTED_ADAPTER_SHA:
            raise RuntimeError(f"Adapter weights SHA mismatch: {w_sha} != {EXPECTED_ADAPTER_SHA}")
        if c_sha != EXPECTED_CONFIG_SHA:
            raise RuntimeError(f"Adapter config SHA mismatch: {c_sha} != {EXPECTED_CONFIG_SHA}")

    def load_model(self):
        if self.model is None:
            self.verify_adapter_hashes()
            _LOGGER.info(f"Loading tokenizer {self.base_model_id} ({self.revision})...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                revision=self.revision,
                trust_remote_code=True,
            )

            _LOGGER.info(f"Loading base model {self.base_model_id} on {self.device}...")
            torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

            if HAS_QWEN35_COND:
                base = Qwen3_5ForConditionalGeneration.from_pretrained(
                    self.base_model_id,
                    revision=self.revision,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            elif self.allow_compatibility_fallback:
                _LOGGER.warning("Using AutoModelForCausalLM fallback in compatibility mode (non-authoritative).")
                from transformers import AutoModelForCausalLM
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    revision=self.revision,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            else:
                raise RuntimeError(
                    "Exact P70 production reproduction requires transformers with native Qwen3_5ForConditionalGeneration "
                    "(canonical reproduction lock: transformers==5.17.0). "
                    "To allow generic causal LM execution on non-authoritative environments, set allow_compatibility_fallback=True."
                )

            if not self.device.startswith("cuda") or ":" in self.device:
                base = base.to(self.device)

            self.model = PeftModel.from_pretrained(
                base,
                str(self.adapter_dir),
            )
            self.model.eval()

            # Verify 192/192 adapter binding
            verify_exact_adapter_binding(self.model, self.adapter_dir, self.device)

    def generate_answer(
        self,
        qid: str,
        question: str,
        packed_evidence: PackedEvidence,
    ) -> LegalAnswer:
        self.load_model()
        prompt = build_generation_prompt(question, packed_evidence.formatted_context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        input_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                **self.decode_config,
            )

        continuation_ids = output_ids[0, input_len:].detach().cpu().tolist()
        raw_tokens_count = len(continuation_ids)
        eos_token_id = self.decode_config.get("eos_token_id", 248046)
        eos_seen = bool(eos_token_id in continuation_ids)
        max_new_tokens = self.decode_config.get("max_new_tokens", 1536)
        raw_cap_hit = compute_raw_cap_hit(raw_tokens_count, max_new_tokens, eos_seen)

        # Apply production duplicate guard (token_suffix_loop_sanitizer)
        cleaned_ids, guard_applied = sanitize_duplicate_loops(continuation_ids, tokenizer=self.tokenizer)
        answer_text = self.tokenizer.decode(cleaned_ids, skip_special_tokens=True).strip()

        return LegalAnswer(
            qid=qid,
            question=question,
            answer=answer_text,
            evidence_ids=packed_evidence.chunk_ids,
            tokens_generated=len(cleaned_ids),
            duplicate_guard_applied=guard_applied,
            metadata={
                "raw_tokens_count": raw_tokens_count,
                "eos_seen": eos_seen,
                "raw_cap_hit": raw_cap_hit,
            },
        )
