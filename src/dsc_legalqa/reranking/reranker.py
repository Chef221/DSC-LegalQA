"""Cross-encoder reranker wrapping Qwen3-Reranker-0.6B."""

from typing import Sequence
import numpy as np
from dsc_legalqa.data.schema import RetrievalHit

RERANKER_MODEL_ID: str = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_PINNED_REVISION: str = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


class NeuralReranker:
    """Reranks Prefix 20 candidates and preserves tail (21..100)."""

    def __init__(
        self,
        model_id: str = RERANKER_MODEL_ID,
        revision: str = RERANKER_PINNED_REVISION,
        prefix_k: int = 20,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.revision = revision
        self.prefix_k = prefix_k
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_id,
                revision=self.revision,
                device=self.device,
            )

    def rerank(self, query: str, candidates: list[RetrievalHit]) -> list[RetrievalHit]:
        if not candidates:
            return []
        prefix = candidates[: self.prefix_k]
        tail = candidates[self.prefix_k :]

        self.load_model()
        pairs = [(query, c.text) for c in prefix]
        scores = self.model.predict(pairs)

        # Sort prefix by cross-encoder score descending
        ranked_prefix = sorted(zip(scores, prefix), key=lambda x: -x[0])

        reranked_hits: list[RetrievalHit] = []
        for rank, (score, hit) in enumerate(ranked_prefix, start=1):
            reranked_hits.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    article_number=hit.article_number,
                    text=hit.text,
                    score=float(score),
                    rank=rank,
                    strategy="RERANKED",
                    metadata={"reranker_score": float(score), "original_rank": hit.rank},
                )
            )

        # Append preserved tail with continuous ranks
        for offset, hit in enumerate(tail, start=len(reranked_hits) + 1):
            reranked_hits.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    article_number=hit.article_number,
                    text=hit.text,
                    score=hit.score,
                    rank=offset,
                    strategy=hit.strategy,
                    metadata={"preserved_tail": True, "original_rank": hit.rank},
                )
            )

        return reranked_hits
