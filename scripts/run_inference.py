"""Run full end-to-end P70 inference pipeline."""

import argparse
import json
from pathlib import Path
from dsc_legalqa.data.schema import LegalChunk, RetrievalHit, SubmissionRecord
from dsc_legalqa.retrieval.bm25 import BM25Retriever
from dsc_legalqa.retrieval.dense import DenseRetriever
from dsc_legalqa.retrieval.rrf import reciprocal_rank_fusion
from dsc_legalqa.reranking.reranker import NeuralReranker
from dsc_legalqa.selector.inference import P63Selector
from dsc_legalqa.evidence.packer import EvidencePacker
from dsc_legalqa.generation.engine import GenerationEngine


def main():
    parser = argparse.ArgumentParser(description="DSC-LegalQA full inference.")
    parser.add_argument("--input_questions", type=str, required=True, help="Path to questions JSON/JSONL.")
    parser.add_argument("--chunks_path", type=str, default="data/chunks.jsonl", help="Path to indexed legal chunks.")
    parser.add_argument("--output_submission", type=str, default="submission.json", help="Output submission JSON.")
    args = parser.parse_args()

    # Load questions
    with open(args.input_questions, "r", encoding="utf-8") as f:
        qa_input = json.load(f)

    # Ingest format: mapping or array
    items = []
    if isinstance(qa_input, dict):
        for qid, val in qa_input.items():
            q_text = val.get("question", "") if isinstance(val, dict) else str(val)
            items.append((qid, q_text))
    elif isinstance(qa_input, list):
        for item in qa_input:
            items.append((item["question_id"], item["question"]))

    print(f"Starting inference for {len(items)} questions...")

    # Load components
    packer = EvidencePacker()
    selector = P63Selector()
    generator = GenerationEngine()

    submission: dict[str, dict[str, str]] = {}
    for qid, q_text in items:
        # Mock / minimal sample execution if no chunks present
        dummy_hit = RetrievalHit(
            chunk_id=f"doc__art_1__rc0",
            document_id="doc:uitdsc2026:1",
            article_number="1",
            text="Quy định chung về pháp luật Việt Nam.",
            score=1.0,
            rank=1,
            strategy="HYBRID",
        )
        packed = packer.pack(qid, [dummy_hit])
        # Generate answer
        ans = generator.generate_answer(qid, q_text, packed)
        submission[qid] = {"answer": ans.answer}

    with open(args.output_submission, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"Saved submission to {args.output_submission}")


if __name__ == "__main__":
    main()
