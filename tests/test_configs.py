"""Test configuration loading and model specifications."""

from pathlib import Path
from dsc_legalqa.utils.config import load_config, validate_parameter_budget


def test_load_config():
    cfg = load_config("configs")
    assert "dense_retriever" in cfg.models
    assert "reranker" in cfg.models
    assert "generator" in cfg.models
    assert "generator_lora" in cfg.models

    assert cfg.retrieval["retrieval"]["bm25"]["k1"] == 1.5
    assert cfg.retrieval["retrieval"]["fusion"]["rrf_constant"] == 10
    assert cfg.selector["selector"]["tau_deploy"] == 0.100
    assert cfg.generation["generation"]["decoding"]["max_new_tokens"] == 1536


def test_model_pinned_revisions():
    cfg = load_config("configs")
    # Dense retriever
    assert cfg.models["dense_retriever"]["model_id"] == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.models["dense_retriever"]["pinned_revision"] == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

    # Reranker
    assert cfg.models["reranker"]["model_id"] == "Qwen/Qwen3-Reranker-0.6B"
    assert cfg.models["reranker"]["pinned_revision"] == "e61197ed45024b0ed8a2d74b80b4d909f1255473"

    # Generator
    assert cfg.models["generator"]["model_id"] == "Qwen/Qwen3.5-2B"
    assert cfg.models["generator"]["pinned_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
