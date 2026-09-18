"""BM25 sparse lexical retrieval implementation."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence
from dsc_legalqa.data.schema import LegalChunk, RetrievalHit


class BM25Retriever:
    """In-memory BM25Okapi retriever for Vietnamese legal chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[LegalChunk] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_freqs: list[Counter[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def fit(self, chunks: Sequence[LegalChunk]) -> "BM25Retriever":
        self.chunks = list(chunks)
        n = len(self.chunks)
        self.doc_len = []
        self.doc_freqs = []
        self.df = Counter()

        total_len = 0
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.search_text or chunk.text)
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for word in freqs:
                self.df[word] += 1

        self.avgdl = total_len / n if n > 0 else 1.0
        # Compute IDF
        self.idf = {}
        for word, count in self.df.items():
            # BM25Okapi idf formula
            self.idf[word] = math.log(1.0 + (n - count + 0.5) / (count + 0.5))
        return self

    def retrieve(self, query: str, top_k: int = 100) -> list[RetrievalHit]:
        q_tokens = self._tokenize(query)
        scores: list[float] = [0.0] * len(self.chunks)

        for token in q_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            for idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                d_len = self.doc_len[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avgdl))
                scores[idx] += token_idf * (numerator / denominator)

        # Rank candidates
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
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
                    strategy="BM25",
                )
            )
        return hits
