"""Sparse, dense, and hybrid retrieval modules."""

from dsc_legalqa.retrieval.bm25 import BM25Retriever
from dsc_legalqa.retrieval.dense import DenseRetriever
from dsc_legalqa.retrieval.rrf import reciprocal_rank_fusion

__all__ = ["BM25Retriever", "DenseRetriever", "reciprocal_rank_fusion"]
