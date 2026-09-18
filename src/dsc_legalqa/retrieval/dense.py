"""Dense vector retriever using Qwen3-Embedding-0.6B."""

from pathlib import Path
from typing import Sequence
import numpy as np
from dsc_legalqa.data.schema import LegalChunk, RetrievalHit

MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B"
PINNED_REVISION: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class DenseRetriever:
    """Dense retriever wrapping Qwen3-Embedding-0.6B encoder."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = PINNED_REVISION,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.model = None
        self.chunks: list[LegalChunk] = []
        self.embeddings: np.ndarray | None = None

    def load_model(self):
        """Lazy load embedding model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_id,
                revision=self.revision,
                device=self.device,
            )

    def index_chunks(self, chunks: Sequence[LegalChunk], embeddings: np.ndarray | None = None):
        self.chunks = list(chunks)
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.load_model()
            texts = [c.search_text or c.text for c in self.chunks]
            self.embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

    def retrieve(self, query: str, top_k: int = 100) -> list[RetrievalHit]:
        if self.embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("DenseRetriever has no indexed chunks.")
        self.load_model()
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity
        scores = np.dot(self.embeddings, q_emb)
        ranked_indices = np.argsort(-scores)[:top_k]

        hits: list[RetrievalHit] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    article_number=chunk.article_number,
                    text=chunk.text,
                    score=float(scores[idx]),
                    rank=rank,
                    strategy="DENSE",
                )
            )
        return hits
