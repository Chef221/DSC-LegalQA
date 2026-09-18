"""Live GPU integration test for Qwen3 retrieval, reranker, and Qwen3.5-2B + P70 generation.

To run: pytest -m gpu
"""

import pytest
import torch
from pathlib import Path


@pytest.mark.gpu
def test_p70_gpu_live_integration():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available; live GPU integration test skipped.")

    from dsc_legalqa.generation.engine import GenerationEngine, verify_exact_adapter_binding
    from dsc_legalqa.data.schema import PackedEvidence

    engine = GenerationEngine(device="cuda:0")
    engine.load_model()

    assert engine.model is not None
    assert engine.tokenizer is not None

    packed = PackedEvidence(
        query_id="Q_GPU_SMOKE",
        chunk_ids=("c1",),
        formatted_context="[E1] Người lao động có quyền đơn phương chấm dứt hợp đồng lao động theo quy định.",
        token_count_estimate=20,
    )

    ans = engine.generate_answer(
        qid="Q_GPU_SMOKE",
        question="Người lao động có quyền đơn phương chấm dứt hợp đồng không?",
        packed_evidence=packed,
    )

    assert ans.answer is not None
    assert len(ans.answer) > 0
    assert ans.tokens_generated > 0
