"""Dense vector retriever using Qwen3-Embedding-0.6B with exact Q2 instruction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union
import unicodedata
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from dsc_legalqa.data.schema import LegalChunk, RetrievalHit

_LOGGER = logging.getLogger(__name__)

MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B"
PINNED_REVISION: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EXPECTED_PARAMS: int = 595_776_512
EMBED_DIM: int = 1024

# Production Q2 query instruction template
DENSE_Q2_TEMPLATE: str = (
    "Instruct: Given a legal question, retrieve relevant legal passages "
    "that provide evidence for answering the question\nQuery:{query}"
)


def normalize_query_text(raw_query: str) -> str:
    """Canonical NFC normalization with collapsed whitespace."""
    cleaned = " ".join(raw_query.strip().split())
    return unicodedata.normalize("NFC", cleaned)


def format_query_instruction(query: str) -> str:
    """Format query string with frozen Q2 legal retrieval instruction template."""
    norm_q = normalize_query_text(query)
    return DENSE_Q2_TEMPLATE.format(query=norm_q)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract embeddings from last non-pad token position according to attention_mask."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class DenseRetriever:
    """Dense retriever wrapping Qwen3-Embedding-0.6B with exact last-token pooling and Q2 instruction."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = PINNED_REVISION,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.model = None
        self.tokenizer = None
        self.chunks: list[LegalChunk] = []
        self.embeddings: np.ndarray | None = None

    def load_model(self):
        """Lazy load Hugging Face Qwen3-Embedding-0.6B model and tokenizer."""
        if self.model is None:
            _LOGGER.info(f"Loading dense embedding model {self.model_id} ({self.revision}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
                padding_side="right",
            )
            self.model = AutoModel.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into normalized 1024-dim dense embeddings using last-token pooling."""
        self.load_model()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**encoded)
                embeds = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                # L2 normalize
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                all_embeddings.append(embeds.cpu().to(torch.float32).numpy())
        return np.vstack(all_embeddings)

    def index_chunks(self, chunks: Sequence[LegalChunk], embeddings: np.ndarray | None = None):
        self.chunks = list(chunks)
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            texts = [c.search_text or c.text for c in self.chunks]
            self.embeddings = self.encode_texts(texts)

    def retrieve(self, query: str, top_k: int = 100) -> list[RetrievalHit]:
        if self.embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("DenseRetriever has no indexed chunks.")
        # Apply Q2 legal query instruction
        formatted_q = format_query_instruction(query)
        q_emb = self.encode_texts([formatted_q])[0]

        # Inner product on L2-normalized embeddings = cosine similarity
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
                    strategy="DENSE_Q2",
                )
            )
        return hits
