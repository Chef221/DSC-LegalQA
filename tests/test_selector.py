"""Test P63 selector features and decision threshold."""

from pathlib import Path
from dsc_legalqa.selector.features import extract_37_features, FEATURE_NAMES
from dsc_legalqa.selector.inference import P63Selector, FROZEN_MODEL_SHA256, FROZEN_TAU
from dsc_legalqa.utils.config import sha256_file


def test_p63_feature_dimensionality():
    assert len(FEATURE_NAMES) == 37
    q = "Thời hạn hợp đồng lao động xác định thời hạn là bao lâu?"
    cands = [
        {"document_id": "doc1", "article_number": "20", "rank": 1, "score": 0.95, "text": "Hợp đồng lao động", "source_branch": "INCUMBENT"},
        {"document_id": "doc1", "article_number": "21", "rank": 2, "score": 0.88, "text": "Nội dung hợp đồng", "source_branch": "WIDER_TOPK_RETRIEVAL"},
    ]
    feats = extract_37_features(q, cands, "Rendered text", "Incumbent text")
    assert len(feats) == 37
    assert isinstance(feats[0], float)


def test_p63_artifact_hashes():
    model_path = Path("artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl")
    meta_path = Path("artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json")

    assert model_path.exists()
    assert meta_path.exists()

    actual_sha = sha256_file(model_path)
    assert actual_sha == FROZEN_MODEL_SHA256
    assert FROZEN_TAU == 0.100


def test_p63_selector_inference():
    selector = P63Selector(
        model_path="artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl",
        meta_path="artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json",
    )
    decision = selector.select(
        question="Hợp đồng lao động có thời hạn bao lâu?",
        incumbent_chunk_ids=("c1",),
        incumbent_text="Incumbent text",
        override_candidates=[],
    )
    assert decision.action == "INCUMBENT"
    assert decision.selected_chunk_ids == ("c1",)
    assert decision.tau_applied == 0.100
