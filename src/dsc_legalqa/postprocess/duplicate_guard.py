"""Production token suffix loop sanitizer (SHA256: 1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527)."""

from typing import Sequence


def sanitize_duplicate_loops(
    token_ids: Sequence[int],
    min_cycle_len: int = 4,
    max_cycle_len: int = 64,
    min_repeats: int = 3,
) -> tuple[list[int], bool]:
    """Detect and sanitize trailing repetition loops in generated token stream.

    If a token pattern repeats min_repeats or more times at the suffix of the output,
    it cuts off the duplicate iterations and leaves only one canonical instance.
    """
    tokens = list(token_ids)
    n = len(tokens)
    if n < min_cycle_len * min_repeats:
        return tokens, False

    # Check for periodic suffix loop
    for cycle_len in range(min_cycle_len, min(max_cycle_len + 1, n // min_repeats + 1)):
        pattern = tokens[-cycle_len:]
        repeats = 1
        pos = n - cycle_len
        while pos >= cycle_len:
            prev_chunk = tokens[pos - cycle_len : pos]
            if prev_chunk == pattern:
                repeats += 1
                pos -= cycle_len
            else:
                break
        if repeats >= min_repeats:
            # Cut suffix to keep only 1 instance of the cycle
            cut_point = pos + cycle_len
            return tokens[:cut_point], True

    return tokens, False
