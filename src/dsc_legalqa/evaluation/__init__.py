"""Official competition metric evaluators."""

from dsc_legalqa.evaluation.scorer import (
    compute_meteor_score,
    compute_rouge_l_score,
    evaluate_predictions,
)

__all__ = ["compute_meteor_score", "compute_rouge_l_score", "evaluate_predictions"]
