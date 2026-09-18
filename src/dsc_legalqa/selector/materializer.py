"""Materializes P63 candidates and upstream actions from retrieval/reranker hits."""

from __future__ import annotations

from typing import Any, Sequence
from dsc_legalqa.data.schema import RetrievalHit


def materialize_candidates_from_hits(
    hits: Sequence[RetrievalHit],
    max_override_candidates: int = 10,
) -> tuple[tuple[str, ...], str, list[dict[str, Any]]]:
    """Extracts (incumbent_chunk_ids, incumbent_text, override_candidates) from reranked hits.

    - Incumbent action: Canonical First 2 hits (or First 1 if only 1 exists).
    - Override candidate pool: Alternative candidate pairs/singletons from Top-K hits.
    """
    if not hits:
        return (), "", []

    # Incumbent: Top-2 hits
    incumbent_hits = list(hits[:2])
    incumbent_chunk_ids = tuple(h.chunk_id for h in incumbent_hits)
    incumbent_text = "\n\n".join(h.text.strip() for h in incumbent_hits)

    # Build override candidates from top reranked hits
    override_candidates: list[dict[str, Any]] = []

    # 1. Single-hit candidates from Top-5
    for rank_idx, h in enumerate(hits[:5], start=1):
        item = {
            "chunk_id": h.chunk_id,
            "document_id": h.document_id,
            "article_number": h.article_number,
            "rank": rank_idx,
            "score": h.score,
            "source_branch": "WIDER_TOPK_RETRIEVAL",
            "text": h.text.strip(),
        }
        override_candidates.append({
            "candidate_id": len(override_candidates),
            "chunk_ids": [h.chunk_id],
            "items": [item],
            "text": h.text.strip(),
            "source_branch": "WIDER_TOPK_RETRIEVAL",
        })

    # 2. Pair candidates from Top-5 (e.g. (0, 2), (1, 2), (0, 3))
    prefix_pool = list(hits[:min(5, len(hits))])
    for i in range(len(prefix_pool)):
        for j in range(i + 1, len(prefix_pool)):
            h1 = prefix_pool[i]
            h2 = prefix_pool[j]
            # Skip if this is the exact incumbent pair
            if (h1.chunk_id, h2.chunk_id) == incumbent_chunk_ids:
                continue

            c1 = {
                "chunk_id": h1.chunk_id,
                "document_id": h1.document_id,
                "article_number": h1.article_number,
                "rank": i + 1,
                "score": h1.score,
                "source_branch": "WIDER_TOPK_RETRIEVAL",
                "text": h1.text.strip(),
            }
            c2 = {
                "chunk_id": h2.chunk_id,
                "document_id": h2.document_id,
                "article_number": h2.article_number,
                "rank": j + 1,
                "score": h2.score,
                "source_branch": "WIDER_TOPK_RETRIEVAL",
                "text": h2.text.strip(),
            }
            pair_text = f"{h1.text.strip()}\n\n{h2.text.strip()}"
            override_candidates.append({
                "candidate_id": len(override_candidates),
                "chunk_ids": [h1.chunk_id, h2.chunk_id],
                "items": [c1, c2],
                "text": pair_text,
                "source_branch": "WIDER_TOPK_RETRIEVAL",
            })
            if len(override_candidates) >= max_override_candidates:
                break
        if len(override_candidates) >= max_override_candidates:
            break

    return incumbent_chunk_ids, incumbent_text, override_candidates
