"""Official competition scorer matching Codabench Task 2 contract."""

import re
from typing import Mapping
from nltk.translate.meteor_score import meteor_score


def _whitespace_tokenize(text: str) -> list[str]:
    return text.strip().split()


def compute_meteor_score(reference: str, hypothesis: str) -> float:
    """Compute NLTK METEOR score on whitespace tokens."""
    ref_tokens = _whitespace_tokenize(reference)
    hyp_tokens = _whitespace_tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    return float(meteor_score([ref_tokens], hyp_tokens))


def _lcs(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l_score(reference: str, hypothesis: str) -> float:
    """Compute standard ROUGE-L F1 score."""
    ref_tokens = _whitespace_tokenize(reference)
    hyp_tokens = _whitespace_tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs_len = _lcs(ref_tokens, hyp_tokens)
    prec = lcs_len / len(hyp_tokens)
    rec = lcs_len / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def evaluate_predictions(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
) -> dict[str, float]:
    """Compute official macro-averaged METEOR and ROUGE-L scores."""
    meteors: list[float] = []
    rouges: list[float] = []

    for qid, ref in references.items():
        hyp = predictions.get(qid, "")
        meteors.append(compute_meteor_score(ref, hyp))
        rouges.append(compute_rouge_l_score(ref, hyp))

    mean_meteor = sum(meteors) / len(meteors) if meteors else 0.0
    mean_rouge = sum(rouges) / len(rouges) if rouges else 0.0

    return {
        "meteor": round(mean_meteor, 9),
        "rouge_l": round(mean_rouge, 9),
        "sample_count": len(meteors),
    }
