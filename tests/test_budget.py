"""Test strict parameter budget compliance (< 4B)."""

from dsc_legalqa.utils.config import load_config, validate_parameter_budget


def test_parameter_budget_less_than_4b():
    cfg = load_config("configs")
    summary = validate_parameter_budget(cfg.models)

    assert summary.dense_retriever_params == 595776512
    assert summary.reranker_params == 595776512
    assert summary.generator_base_params == 2213241664
    assert summary.lora_trainable_params == 21823488

    assert summary.total_system_params == 3426618176
    assert summary.total_system_params < 4000000000
    assert summary.is_compliant is True
    assert summary.headroom_params > 500000000
    assert summary.headroom_percentage >= 12.0
