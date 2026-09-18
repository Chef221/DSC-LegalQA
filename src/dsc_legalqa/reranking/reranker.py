"""Production Qwen3-Reranker-0.6B Prefix20 causal-LM logit-difference reranker."""

from __future__ import annotations

import logging
import math
from typing import Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dsc_legalqa.data.schema import RetrievalHit

_LOGGER = logging.getLogger(__name__)

RERANKER_MODEL_ID: str = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_PINNED_REVISION: str = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RERANKER_ORIGINAL_PARAMETERS: int = 595_776_512
RERANKER_MODEL_CONTEXT_LIMIT: int = 32_768
RERANKER_INSTRUCTION: str = "Given the user query, retrieval the relevant passages"

TOKEN_TRUE_ID: int = 9693   # "yes"
TOKEN_FALSE_ID: int = 2152  # "no"
TOKEN_TRUE_STR: str = "yes"
TOKEN_FALSE_STR: str = "no"

PREFIX_RERANK_DEPTH: int = 20
TOTAL_CANDIDATE_DEPTH: int = 100


def format_reranker_pair(
    query: str,
    document: str,
    instruction: str = RERANKER_INSTRUCTION,
) -> str:
    """Format a query-document pair according to the exact official Qwen3 reranker chat template."""
    q = '"'
    return (
        f"<|im_start|>system\n"
        f"Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        f"Note that the answer can only be {q}yes{q} or {q}no{q}.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n\n"
        f"</think>\n\n"
    )


def compute_raw_reranker_score(true_logit: float, false_logit: float) -> float:
    """Compute raw reranker score as true_logit(9693) - false_logit(2152)."""
    diff = float(true_logit) - float(false_logit)
    if not math.isfinite(diff):
        raise ValueError(f"Non-finite reranker score difference: true={true_logit}, false={false_logit}")
    return diff


class NeuralReranker:
    """Reranks Prefix 20 candidates via Qwen3 causal-LM yes/no logit difference and preserves tail (21..100)."""

    def __init__(
        self,
        model_id: str = RERANKER_MODEL_ID,
        revision: str = RERANKER_PINNED_REVISION,
        prefix_k: int = PREFIX_RERANK_DEPTH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_id = model_id
        self.revision = revision
        self.prefix_k = prefix_k
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            _LOGGER.info(f"Loading reranker model {self.model_id} ({self.revision}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                padding_side="left",
                trust_remote_code=True,
            )

            # Validate token IDs
            yes_ids = self.tokenizer.encode(TOKEN_TRUE_STR, add_special_tokens=False)
            no_ids = self.tokenizer.encode(TOKEN_FALSE_STR, add_special_tokens=False)
            if not yes_ids or yes_ids[-1] != TOKEN_TRUE_ID:
                raise RuntimeError(f"Reranker YES token ID mismatch: expected {TOKEN_TRUE_ID}, got {yes_ids}")
            if not no_ids or no_ids[-1] != TOKEN_FALSE_ID:
                raise RuntimeError(f"Reranker NO token ID mismatch: expected {TOKEN_FALSE_ID}, got {no_ids}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            self.model.config.use_cache = False

    def rerank(self, query: str, candidates: list[RetrievalHit]) -> list[RetrievalHit]:
        if not candidates:
            return []
        prefix = candidates[: self.prefix_k]
        tail = candidates[self.prefix_k :]

        self.load_model()
        scores: list[float] = []
        for cand in prefix:
            prompt_text = format_reranker_pair(query, cand.text)
            enc = self.tokenizer(prompt_text, add_special_tokens=False, truncation=False, return_tensors="pt")
            input_len = int(enc["input_ids"].shape[-1])
            if input_len > RERANKER_MODEL_CONTEXT_LIMIT:
                raise RuntimeError(f"Pair input tokens ({input_len}) exceeds limit ({RERANKER_MODEL_CONTEXT_LIMIT})")
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
                logits = outputs.logits[:, -1, :]
                true_logit = float(logits[:, TOKEN_TRUE_ID].float().cpu())
                false_logit = float(logits[:, TOKEN_FALSE_ID].float().cpu())
            scores.append(compute_raw_reranker_score(true_logit, false_logit))

        # Sort prefix descending by reranker raw score, tie-breaking by original rank ASC
        scored_prefix = list(zip(scores, prefix))
        scored_prefix.sort(key=lambda item: (-item[0], item[1].rank))

        reranked_hits: list[RetrievalHit] = []
        for rank, (score, hit) in enumerate(scored_prefix, start=1):
            meta = dict(hit.metadata or {})
            meta["reranker_score"] = float(score)
            meta["original_fused_rank"] = hit.rank
            meta["reranker_scored"] = True
            reranked_hits.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    article_number=hit.article_number,
                    text=hit.text,
                    score=float(score),
                    rank=rank,
                    strategy="QWEN3_PREFIX20_RERANKED",
                    metadata=meta,
                )
            )

        # Preserved tail (ranks 21..100) exactly in original fused order
        for offset, hit in enumerate(tail, start=len(reranked_hits) + 1):
            meta = dict(hit.metadata or {})
            meta["original_fused_rank"] = hit.rank
            meta["reranker_scored"] = False
            meta["preserved_tail"] = True
            reranked_hits.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    article_number=hit.article_number,
                    text=hit.text,
                    score=hit.score,
                    rank=offset,
                    strategy=hit.strategy,
                    metadata=meta,
                )
            )

        return reranked_hits
