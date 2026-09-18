"""Production duplicate guard wrapper around frozen token_suffix_loop_sanitizer.

Authoritative duplicate sanitizer SHA256:
1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from dsc_legalqa.postprocess.token_suffix_loop_sanitizer import (
    find_best_run,
    sanitize_tokens,
)

FROZEN_GUARD_SHA256 = "1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527"
FROZEN_GUARD_BYTES = 26461


class _FallbackTokenizer:
    """Fallback tokenizer for decoding debug blocks when no HuggingFace tokenizer is provided."""
    def decode(self, token_ids: Sequence[int]) -> str:
        return f"[tokens:{len(token_ids)}]"


def verify_guard_provenance() -> bool:
    """Verify that token_suffix_loop_sanitizer.py exactly matches production authority bytes."""
    guard_path = Path(__file__).parent / "token_suffix_loop_sanitizer.py"
    if not guard_path.is_file():
        raise FileNotFoundError(f"Missing production duplicate guard: {guard_path}")
    data = guard_path.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    if h != FROZEN_GUARD_SHA256:
        raise RuntimeError(
            f"Production duplicate guard hash mismatch: expected {FROZEN_GUARD_SHA256}, got {h}"
        )
    return True


def sanitize_duplicate_loops(
    token_ids: Sequence[int],
    tokenizer: Any = None,
) -> tuple[list[int], bool]:
    """Sanitize generated token stream using the exact frozen production detector and transform.

    Parameters
    ----------
    token_ids : Sequence[int]
        Raw continuation token IDs from Qwen3.5 generation.
    tokenizer : Any, optional
        HF tokenizer with .decode() for diagnostic run metadata recording.

    Returns
    -------
    tuple[list[int], bool]
        Cleaned token ID list and boolean indicating whether loop sanitizer triggered.
    """
    tok = tokenizer if tokenizer is not None else _FallbackTokenizer()
    cleaned_tokens, runs_removed = sanitize_tokens(list(token_ids), tok)
    guard_applied = len(runs_removed) > 0
    return cleaned_tokens, guard_applied


__all__ = [
    "FROZEN_GUARD_SHA256",
    "FROZEN_GUARD_BYTES",
    "find_best_run",
    "sanitize_tokens",
    "sanitize_duplicate_loops",
    "verify_guard_provenance",
]
