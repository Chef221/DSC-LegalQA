"""Deterministic end-to-end smoke test using tiny synthetic fixtures."""

from dsc_legalqa.data.schema import LegalChunk, LegalDocument
from dsc_legalqa.retrieval.bm25 import BM25Retriever
from dsc_legalqa.retrieval.rrf import reciprocal_rank_fusion
from dsc_legalqa.evidence.packer import EvidencePacker
from dsc_legalqa.selector.inference import P63Selector
from dsc_legalqa.evaluation.scorer import compute_meteor_score, compute_rouge_l_score


def test_end_to_end_smoke_flow():
    # 1. Synthetic Corpus
    chunks = [
        LegalChunk("c1", "doc1", "1", "1", "Người lao động được nghỉ làm việc, hưởng nguyên lương trong ngày Quốc khánh.", "người lao động nghỉ quốc khánh hưởng nguyên lương"),
        LegalChunk("c2", "doc1", "2", "2", "Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.", "thời giờ làm việc bình thường 8 giờ 48 giờ"),
        LegalChunk("c3", "doc2", "3", "3", "Người sử dụng lao động có quyền tạm đình chỉ công việc của người lao động khi vụ việc có những tình tiết phức tạp.", "tạm đình chỉ công việc tình tiết phức tạp"),
    ]

    # 2. BM25 Retrieval
    bm25 = BM25Retriever()
    bm25.fit(chunks)
    query = "Người lao động được nghỉ ngày Quốc khánh như thế nào?"
    hits_bm25 = bm25.retrieve(query, top_k=2)
    assert len(hits_bm25) > 0
    assert hits_bm25[0].chunk_id == "c1"

    # 3. Equal RRF
    fused = reciprocal_rank_fusion(hits_bm25, hits_bm25, rrf_constant=10, top_k=2)
    assert len(fused) > 0

    # 4. Evidence Packing
    packer = EvidencePacker()
    packed = packer.pack("Q_TEST_01", fused[:2])
    assert "[E1]" in packed.formatted_context

    # 5. P63 Selector
    selector = P63Selector(
        model_path="artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl",
        meta_path="artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json",
    )
    decision = selector.select(query, packed.chunk_ids, packed.formatted_context, [])
    assert decision.action == "INCUMBENT"

    # 6. Evaluation Scorer
    ref = "Người lao động được nghỉ làm việc và hưởng nguyên lương trong ngày lễ Quốc khánh."
    hyp = "Người lao động được nghỉ làm việc, hưởng nguyên lương trong ngày Quốc khánh."
    meteor = compute_meteor_score(ref, hyp)
    rouge = compute_rouge_l_score(ref, hyp)

    assert meteor > 0.8
    assert rouge > 0.8
