"""Core immutable data classes for LegalQA pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LegalDocument:
    """Official organizer context item."""
    id: int | str
    name: str
    link: str
    passage: str


@dataclass(frozen=True, slots=True)
class LegalChunk:
    """Processed retrieval chunk with hierarchy metadata."""
    chunk_id: str
    document_id: str
    article_number: str
    clause_number: str
    text: str
    search_text: str
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """Ranked hit from sparse or dense retrieval branch."""
    chunk_id: str
    document_id: str
    article_number: str
    text: str
    score: float
    rank: int
    strategy: str  # BM25, DENSE, HYBRID, RERANKED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PackedEvidence:
    """Packed evidence context ready for generator input."""
    query_id: str
    chunk_ids: tuple[str, ...]
    formatted_context: str
    token_count_estimate: int = 0


@dataclass(frozen=True, slots=True)
class LegalAnswer:
    """Generated prose answer with telemetry."""
    qid: str
    question: str
    answer: str
    evidence_ids: tuple[str, ...]
    insufficient_evidence: bool = False
    tokens_generated: int = 0
    duplicate_guard_applied: bool = False


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """Codabench submission record format: qid -> {"answer": text}."""
    qid: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return {"answer": self.answer}
