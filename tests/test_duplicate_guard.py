"""Tests for frozen production duplicate guard (token_suffix_loop_sanitizer)."""

import hashlib
from pathlib import Path
from dsc_legalqa.postprocess.duplicate_guard import (
    FROZEN_GUARD_SHA256,
    FROZEN_GUARD_BYTES,
    verify_guard_provenance,
    find_best_run,
    sanitize_duplicate_loops,
)


def test_duplicate_guard_provenance():
    """Ensure token_suffix_loop_sanitizer.py exactly matches accepted P70 authority SHA256."""
    guard_path = Path("src/dsc_legalqa/postprocess/token_suffix_loop_sanitizer.py")
    assert guard_path.is_file(), "token_suffix_loop_sanitizer.py missing"
    data = guard_path.read_bytes()
    assert len(data) == FROZEN_GUARD_BYTES, f"Bytes mismatch: {len(data)} != {FROZEN_GUARD_BYTES}"
    actual_sha = hashlib.sha256(data).hexdigest()
    assert actual_sha == FROZEN_GUARD_SHA256, f"SHA mismatch: {actual_sha} != {FROZEN_GUARD_SHA256}"
    assert verify_guard_provenance() is True


def test_no_loop():
    """Short or non-repeating token sequences must pass through untouched."""
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cleaned, applied = sanitize_duplicate_loops(tokens)
    assert cleaned == tokens
    assert applied is False


def test_production_loop_detector_and_sanitization():
    """Verify that a suffix loop with L=5, R=5 (run_len=25 >= 24, coverage >= 15%) is detected and cleaned."""
    # Prefix of 10 tokens, followed by 5 repeats of a 5-token block (total tokens = 35)
    # run length = 25 >= 24, coverage = 25 / 35 = 71.4% >= 15%
    prefix = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    block = [101, 102, 103, 104, 105]
    tokens = prefix + block * 5

    run = find_best_run(tokens)
    assert run is not None
    assert run["L"] == 5
    assert run["R"] == 5
    assert run["run_length"] == 25
    assert run["removable"] == 20  # (5 - 1) * 5

    cleaned, applied = sanitize_duplicate_loops(tokens)
    assert applied is True
    # Retain exactly the first copy of the block
    assert cleaned == prefix + block
