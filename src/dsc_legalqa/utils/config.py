"""Configuration management and parameter-budget validation."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any
import yaml

COMPETITION_PARAMETER_LIMIT: int = 4_000_000_000  # < 4B strict limit


@dataclass
class ModelSpec:
    model_id: str
    pinned_revision: str
    parameter_count: int
    license: str
    role: str
    source_url: str


@dataclass
class ParameterBudgetSummary:
    dense_retriever_params: int
    reranker_params: int
    generator_base_params: int
    lora_trainable_params: int
    total_system_params: int
    competition_limit: int
    headroom_params: int
    headroom_percentage: float
    is_compliant: bool


@dataclass
class AppConfig:
    models: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    selector: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path | str = "configs") -> AppConfig:
    base = Path(config_dir)
    return AppConfig(
        models=load_yaml(base / "models.yaml"),
        retrieval=load_yaml(base / "retrieval.yaml"),
        selector=load_yaml(base / "selector.yaml"),
        generation=load_yaml(base / "generation.yaml"),
    )


def validate_parameter_budget(models_cfg: dict[str, Any]) -> ParameterBudgetSummary:
    """Audit system parameter budget against official UIT DSC 2026 rule (< 4B)."""
    retriever_params = int(models_cfg.get("dense_retriever", {}).get("parameter_count", 595776512))
    reranker_params = int(models_cfg.get("reranker", {}).get("parameter_count", 595776512))
    generator_params = int(models_cfg.get("generator", {}).get("parameter_count", 2213241664))
    lora_params = int(models_cfg.get("generator_lora", {}).get("parameter_count", 21823488))

    total = retriever_params + reranker_params + generator_params + lora_params
    headroom = COMPETITION_PARAMETER_LIMIT - total
    headroom_pct = (headroom / COMPETITION_PARAMETER_LIMIT) * 100.0

    compliant = total < COMPETITION_PARAMETER_LIMIT

    return ParameterBudgetSummary(
        dense_retriever_params=retriever_params,
        reranker_params=reranker_params,
        generator_base_params=generator_params,
        lora_trainable_params=lora_params,
        total_system_params=total,
        competition_limit=COMPETITION_PARAMETER_LIMIT,
        headroom_params=headroom,
        headroom_percentage=round(headroom_pct, 2),
        is_compliant=compliant,
    )


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
