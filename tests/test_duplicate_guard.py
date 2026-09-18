"""Test duplicate loop repetition guard."""

from dsc_legalqa.postprocess.duplicate_guard import sanitize_duplicate_loops


def test_no_loop():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cleaned, applied = sanitize_duplicate_loops(tokens)
    assert cleaned == tokens
    assert applied is False


def test_suffix_repetition_loop():
    # Loop pattern [100, 101, 102, 103] repeated 4 times at end
    prefix = [10, 20, 30]
    cycle = [100, 101, 102, 103]
    tokens = prefix + cycle * 4
    cleaned, applied = sanitize_duplicate_loops(tokens, min_cycle_len=4, min_repeats=3)
    assert applied is True
    # Should cut duplicate repetitions and leave exactly 1 instance of the cycle
    assert cleaned == prefix + cycle
