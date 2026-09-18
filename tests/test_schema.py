"""Test data schemas and contracts."""

from dsc_legalqa.data.schema import (
    LegalDocument,
    LegalChunk,
    RetrievalHit,
    PackedEvidence,
    SubmissionRecord,
)


def test_legal_document_schema():
    doc = LegalDocument(id=1, name="Luat Lao Dong", link="https://example.com", passage="Noi dung luat")
    assert doc.id == 1
    assert doc.name == "Luat Lao Dong"


def test_retrieval_hit_schema():
    hit = RetrievalHit(
        chunk_id="doc1__art_1__rc0",
        document_id="doc1",
        article_number="1",
        text="Noi dung dieu 1",
        score=0.85,
        rank=1,
        strategy="BM25",
    )
    assert hit.rank == 1
    assert hit.strategy == "BM25"


def test_submission_record_shape():
    rec = SubmissionRecord(qid="Q100", answer="Tra loi chi tiet.")
    assert rec.to_dict() == {"answer": "Tra loi chi tiet."}
