"""Test end-to-end production runner wiring and candidate materialization."""

from pathlib import Path
from dsc_legalqa.data.schema import RetrievalHit
from dsc_legalqa.selector.materializer import materialize_candidates_from_hits
from dsc_legalqa.selector.inference import P63Selector


def test_runner_source_wiring_no_mocks():
    """Verify run_inference.py has zero mock/dummy implementations."""
    repo_root = Path(__file__).resolve().parent.parent
    runner_file = repo_root / "scripts" / "run_inference.py"
    assert runner_file.is_file()

    code = runner_file.read_text(encoding="utf-8")

    # Hard assertions against mock fallbacks
    assert "dummy_hit" not in code, "Found hard-coded dummy_hit in run_inference.py!"
    assert "Mock / minimal sample execution" not in code, "Found mock execution banner in run_inference.py!"

    # Positive assertions that real production components are wired
    assert "normalize_query" in code
    assert "BM25Retriever" in code
    assert "DenseRetriever" in code
    assert "reciprocal_rank_fusion" in code
    assert "NeuralReranker" in code
    assert "materialize_candidates_from_hits" in code
    assert "P63Selector" in code
    assert "EvidencePacker" in code
    assert "GenerationEngine" in code
    assert "submission" in code


def test_candidate_materialization_and_p63_decision():
    """Test candidate materializer and both INCUMBENT and OVERRIDE decision logic."""
    hits = [
        RetrievalHit("c1", "doc1", "1", "Quy định về thời hạn hợp đồng.", 2.5, 1, "RERANKED"),
        RetrievalHit("c2", "doc1", "2", "Quy định về thử việc.", 2.1, 2, "RERANKED"),
        RetrievalHit("c3", "doc2", "10", "Tiền lương làm thêm giờ.", 1.8, 3, "RERANKED"),
        RetrievalHit("c4", "doc2", "11", "Nghỉ phép hằng năm.", 1.5, 4, "RERANKED"),
    ]

    inc_cids, inc_text, override_cands = materialize_candidates_from_hits(hits, max_override_candidates=5)

    assert inc_cids == ("c1", "c2")
    assert "Quy định về thời hạn hợp đồng." in inc_text
    assert len(override_cands) > 0

    # Test P63Selector with real model
    selector = P63Selector(
        model_path="artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl",
        meta_path="artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json",
    )

    # When no overrides supplied -> INCUMBENT
    decision_empty = selector.select("Thời hạn hợp đồng là gì?", inc_cids, inc_text, [])
    assert decision_empty.action == "INCUMBENT"
    assert decision_empty.override_applied is False
    assert decision_empty.tau_applied == 0.100

    # When override candidates are supplied -> evaluate HGB prediction against tau=0.100
    decision = selector.select("Thời hạn hợp đồng là gì?", inc_cids, inc_text, override_cands)
    assert decision.action in ("INCUMBENT", "OVERRIDE")
    assert decision.tau_applied == 0.100
    if decision.action == "OVERRIDE":
        assert decision.override_applied is True
        assert decision.predicted_delta >= 0.100
    else:
        assert decision.override_applied is False
        assert decision.predicted_delta < 0.100
