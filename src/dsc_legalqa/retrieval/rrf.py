"""Equal Reciprocal Rank Fusion (RRF) combiner."""

from dataclasses import dataclass
from dsc_legalqa.data.schema import RetrievalHit


@dataclass
class _CandidateRRF:
    hit: RetrievalHit
    bm25_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(
    bm25_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    rrf_constant: int = 10,
    top_k: int = 100,
) -> list[RetrievalHit]:
    """Deterministically fuse BM25 and Dense candidate ranks using equal RRF (k=10).

    RRF formula:
        score = 1.0 / (k + bm25_rank) + 1.0 / (k + dense_rank)
    """
    if rrf_constant <= 0 or top_k <= 0:
        raise ValueError("rrf_constant and top_k must be positive integers.")

    candidates: dict[str, _CandidateRRF] = {}

    for hit in bm25_hits:
        if hit.chunk_id not in candidates:
            candidates[hit.chunk_id] = _CandidateRRF(hit=hit, bm25_rank=hit.rank)
        else:
            candidates[hit.chunk_id].bm25_rank = hit.rank

    for hit in dense_hits:
        if hit.chunk_id not in candidates:
            candidates[hit.chunk_id] = _CandidateRRF(hit=hit, dense_rank=hit.rank)
        else:
            candidates[hit.chunk_id].dense_rank = hit.rank

    scored: list[tuple[float, str, _CandidateRRF]] = []
    for chunk_id, cand in candidates.items():
        score = 0.0
        if cand.bm25_rank is not None:
            score += 1.0 / (rrf_constant + cand.bm25_rank)
        if cand.dense_rank is not None:
            score += 1.0 / (rrf_constant + cand.dense_rank)
        scored.append((score, chunk_id, cand))

    # Sort descending by score, tie-break on chunk_id lexicographically
    scored.sort(key=lambda item: (-item[0], item[1]))

    fused: list[RetrievalHit] = []
    for rank, (score, _, cand) in enumerate(scored[:top_k], start=1):
        fused.append(
            RetrievalHit(
                chunk_id=cand.hit.chunk_id,
                document_id=cand.hit.document_id,
                article_number=cand.hit.article_number,
                text=cand.hit.text,
                score=score,
                rank=rank,
                strategy="HYBRID",
                metadata={
                    "bm25_rank": cand.bm25_rank,
                    "dense_rank": cand.dense_rank,
                    "rrf_score": score,
                },
            )
        )
    return fused
