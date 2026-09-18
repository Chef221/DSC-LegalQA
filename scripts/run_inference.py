"""Full end-to-end production inference runner for UIT DSC 2026 Task 2.

Exact production pipeline:
  Vietnamese legal question
          ↓
  Exact question normalization (NFC + whitespace)
          ↓
  BM25 sparse retrieval (k1=1.5, b=0.75) + Qwen3 dense retrieval (Q2 instruction, last-token pool)
          ↓
  Equal RRF k=10, Top100
          ↓
  Qwen3 causal-LM Prefix20 reranker + preserved ranks 21..100 tail
          ↓
  P63 candidate materialization & 37 reference-free feature extraction
          ↓
  P63 HistGradientBoostingRegressor inference (tau = 0.100)
          ↓
  INCUMBENT / OVERRIDE action selection
          ↓
  Evidence packing ([E1]...[E2]...) & [ANSWER_CONTROL] header
          ↓
  Qwen3.5-2B + P70 LoRA greedy decoding (max_new_tokens=1536)
          ↓
  Exact production duplicate guard (token_suffix_loop_sanitizer)
          ↓
  submission.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import sys
from pathlib import Path


import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Any
import torch

from dsc_legalqa.data.loader import load_legal_chunks
from dsc_legalqa.data.schema import LegalChunk, RetrievalHit
from dsc_legalqa.retrieval.bm25 import BM25Retriever
from dsc_legalqa.retrieval.dense import DenseRetriever
from dsc_legalqa.retrieval.rrf import reciprocal_rank_fusion
from dsc_legalqa.reranking.reranker import NeuralReranker
from dsc_legalqa.selector.materializer import materialize_candidates_from_hits
from dsc_legalqa.selector.inference import P63Selector
from dsc_legalqa.evidence.packer import EvidencePacker
from dsc_legalqa.generation.engine import GenerationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("run_inference")


def normalize_query(raw_query: str) -> str:
    """Canonical NFC normalization with collapsed whitespace."""
    cleaned = " ".join(raw_query.strip().split())
    return unicodedata.normalize("NFC", cleaned)


def main():
    parser = argparse.ArgumentParser(description="DSC-LegalQA full vNext + P63 + P70 production inference.")
    parser.add_argument("--input_questions", type=str, required=True, help="Path to input questions JSON/JSONL.")
    parser.add_argument("--chunks_path", type=str, required=True, help="Path to indexed legal chunks JSON/JSONL.")
    parser.add_argument("--output_submission", type=str, default="submission.json", help="Output submission JSON.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--adapter_dir", type=str, default="artifacts/p70_adapter", help="P70 LoRA adapter directory.")
    parser.add_argument("--p63_model_path", type=str, default="artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl", help="P63 HGB model path.")
    parser.add_argument("--p63_meta_path", type=str, default="artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json", help="P63 metadata path.")
    parser.add_argument("--dense_embeddings_path", type=str, default=None, help="Optional precomputed corpus embeddings .npy.")
    args = parser.parse_args()

    # 1. Load questions
    _LOGGER.info(f"Loading questions from {args.input_questions}...")
    with open(args.input_questions, "r", encoding="utf-8") as f:
        raw_qa = json.load(f)

    items: list[tuple[str, str]] = []
    if isinstance(raw_qa, dict):
        for qid, val in raw_qa.items():
            q_text = val.get("question", "") if isinstance(val, dict) else str(val)
            items.append((str(qid), q_text))
    elif isinstance(raw_qa, list):
        for item in raw_qa:
            items.append((str(item.get("question_id", item.get("id", ""))), item["question"]))

    if not items:
        raise ValueError(f"No questions loaded from {args.input_questions}")
    _LOGGER.info(f"Loaded {len(items)} questions for inference.")

    # 2. Load corpus chunks - fail closed if missing
    _LOGGER.info(f"Loading legal chunks from {args.chunks_path}...")
    chunks = load_legal_chunks(args.chunks_path)
    _LOGGER.info(f"Loaded {len(chunks)} legal chunks.")

    # 3. Initialize components
    _LOGGER.info(f"Initializing BM25 sparse index on {len(chunks)} chunks...")
    bm25 = BM25Retriever(k1=1.5, b=0.75).fit(chunks)

    _LOGGER.info(f"Initializing Dense Qwen3 retriever on {args.device}...")
    dense = DenseRetriever(device=args.device)
    if args.dense_embeddings_path and Path(args.dense_embeddings_path).is_file():
        import numpy as np
        _LOGGER.info(f"Loading precomputed dense embeddings from {args.dense_embeddings_path}...")
        precomputed = np.load(args.dense_embeddings_path)
        dense.index_chunks(chunks, embeddings=precomputed)
    else:
        _LOGGER.info("Encoding corpus chunks with Dense Qwen3 embedding model...")
        dense.index_chunks(chunks)

    _LOGGER.info(f"Initializing Qwen3 Prefix20 reranker on {args.device}...")
    reranker = NeuralReranker(device=args.device)

    _LOGGER.info(f"Initializing P63 evidence selector from {args.p63_model_path}...")
    selector = P63Selector(model_path=args.p63_model_path, meta_path=args.p63_meta_path)
    packer = EvidencePacker()

    _LOGGER.info(f"Initializing Qwen3.5-2B + P70 LoRA generator on {args.device}...")
    generator = GenerationEngine(adapter_dir=args.adapter_dir, device=args.device)

    # Chunk ID map for quick lookup
    chunk_by_id = {c.chunk_id: c for c in chunks}

    # 4. Run end-to-end inference
    submission: dict[str, dict[str, str]] = {}
    _LOGGER.info("Starting execution of production pipeline across all questions...")

    for idx, (qid, raw_question) in enumerate(items, start=1):
        # Step A: Query normalization
        norm_question = normalize_query(raw_question)

        # Step B: Sparse & Dense retrieval (Top100 each)
        bm25_hits = bm25.retrieve(norm_question, top_k=100)
        dense_hits = dense.retrieve(norm_question, top_k=100)

        # Step C: Equal RRF k=10 fusion -> Top100
        fused_hits = reciprocal_rank_fusion(bm25_hits, dense_hits, rrf_k=10, top_k=100)

        # Step D: Qwen3 causal-LM logit-difference reranker on Prefix20 + preserved tail 21..100
        reranked_hits = reranker.rerank(norm_question, fused_hits)

        # Step E: P63 candidate materialization & 37-feature extraction
        incumbent_cids, incumbent_text, override_candidates = materialize_candidates_from_hits(reranked_hits)

        # Step F: P63 HistGradientBoostingRegressor inference (tau = 0.100)
        decision = selector.select(
            question=norm_question,
            incumbent_chunk_ids=incumbent_cids,
            incumbent_text=incumbent_text,
            override_candidates=override_candidates,
        )

        # Step G: Select final evidence hits
        selected_hits: list[RetrievalHit] = []
        hit_map = {h.chunk_id: h for h in reranked_hits}
        for cid in decision.selected_chunk_ids:
            if cid in hit_map:
                selected_hits.append(hit_map[cid])
            elif cid in chunk_by_id:
                c = chunk_by_id[cid]
                selected_hits.append(
                    RetrievalHit(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        article_number=c.article_number,
                        text=c.text,
                        score=0.0,
                        rank=999,
                        strategy="FALLBACK",
                    )
                )

        if not selected_hits:
            # Fallback to top-1 reranked hit if empty
            selected_hits = reranked_hits[:1]

        # Step H: Evidence packing ([E1]...[E2]...)
        packed_evidence = packer.pack(qid, selected_hits[:3])

        # Step I: Generation + exact production duplicate guard (token_suffix_loop_sanitizer)
        answer_result = generator.generate_answer(qid, norm_question, packed_evidence)

        submission[qid] = {"answer": answer_result.answer}

        if idx % 50 == 0 or idx == len(items):
            _LOGGER.info(f"Processed {idx}/{len(items)} questions (Latest: QID {qid}, action={decision.action}, tokens={answer_result.tokens_generated}).")

    # 5. Output submission JSON in official format
    out_path = Path(args.output_submission)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    _LOGGER.info(f"[SUCCESS] Final submission written to {out_path} ({len(submission)} entries).")


if __name__ == "__main__":
    main()
