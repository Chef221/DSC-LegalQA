"""Test Equal Reciprocal Rank Fusion determinism and logic."""

import pytest
from dsc_legalqa.data.schema import RetrievalHit
from dsc_legalqa.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_scoring_and_ranking():
    bm25 = [
        RetrievalHit("c1", "d1", "1", "chunk 1", 10.0, 1, "BM25"),
        RetrievalHit("c2", "d1", "2", "chunk 2", 8.0, 2, "BM25"),
        RetrievalHit("c3", "d2", "3", "chunk 3", 6.0, 3, "BM25"),
    ]
    dense = [
        RetrievalHit("c2", "d1", "2", "chunk 2", 0.9, 1, "DENSE"),
        RetrievalHit("c1", "d1", "1", "chunk 1", 0.8, 2, "DENSE"),
        RetrievalHit("c4", "d3", "4", "chunk 4", 0.7, 3, "DENSE"),
    ]

    fused = reciprocal_rank_fusion(bm25, dense, rrf_constant=10, top_k=10)

    # c1 score: 1/(10+1) + 1/(10+2) = 1/11 + 1/12 = 0.090909 + 0.083333 = 0.174242
    # c2 score: 1/(10+2) + 1/(10+1) = 1/12 + 1/11 = 0.174242
    # Tie-break on chunk_id: c1 < c2
    assert len(fused) == 4
    assert fused[0].chunk_id == "c1"
    assert fused[1].chunk_id == "c2"
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[0].strategy == "HYBRID"


def test_rrf_invalid_args():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], rrf_constant=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], top_k=-1)
