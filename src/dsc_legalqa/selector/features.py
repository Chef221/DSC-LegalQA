"""Reference-free 37-feature extractor for P63 evidence selection."""

import re
import unicodedata
from typing import Any

FEATURE_NAMES = [
    "cardinality", "has_incumbent", "has_wider", "has_non_article", "is_hybrid_wider_na",
    "num_distinct_branches", "is_same_doc", "is_article_pair", "is_non_article_pair",
    "min_rank", "max_rank", "mean_rank", "max_recip_rank", "min_recip_rank",
    "is_top1", "is_top3", "is_top5", "is_top10", "appended_min_rank", "max_orig_score", "min_orig_score",
    "q_char_len", "q_word_count", "cand_char_len", "cand_word_count", "char_len_ratio", "word_len_ratio",
    "token_overlap_count", "token_overlap_ratio", "number_overlap_count", "art_mentioned",
    "legal_kw_count", "legal_kw_density", "has_dieu_heading", "pair_jaccard", "pair_len_diff", "inc_overlap",
]

LEGAL_KEYWORDS = {
    "điều", "khoản", "điểm", "nghị", "định", "luật", "thông", "tư",
    "bộ", "quyết", "pháp", "lệnh", "phạt", "tiền", "thời", "hạn",
    "nghĩa", "vụ", "quyền", "trách", "nhiệm", "hợp", "đồng", "lao", "động",
    "bảo", "hiểm", "xử", "lý", "vi", "phạm", "hành", "chính", "bồi", "thường",
    "chấm", "dứt", "thôi", "việc", "kỷ", "luật", "cơ", "quan", "thẩm", "quyền",
}

NUMBER_REGEX = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def extract_37_features(
    question: str,
    candidate_items: list[dict[str, Any]],
    rendered_text: str,
    incumbent_text: str = "",
) -> list[float]:
    """Extract exactly 37 reference-free features matching P63 schema."""
    cardinality = len(candidate_items)
    assert cardinality in (1, 2), f"Expected cardinality 1 or 2, got {cardinality}"

    c1 = candidate_items[0]
    c2 = candidate_items[1] if cardinality == 2 else None

    # Family 1: Provenance & Composition
    has_incumbent = 1.0 if any("INCUMBENT" in c.get("source_branch", "") for c in candidate_items) else 0.0
    has_wider = 1.0 if any(c.get("source_branch") == "WIDER_TOPK_RETRIEVAL" for c in candidate_items) else 0.0
    has_non_article = 1.0 if any(c.get("source_branch") == "NON_ARTICLE_FALLBACK" for c in candidate_items) else 0.0
    is_hybrid_wider_na = 1.0 if (has_wider and has_non_article) else 0.0
    branches = {c.get("source_branch", "") for c in candidate_items}
    num_distinct_branches = float(len(branches))

    # Family 2: Pair Structure
    if cardinality == 2 and c2 is not None:
        is_same_doc = 1.0 if c1.get("document_id") == c2.get("document_id") else 0.0
        is_article_pair = 1.0 if (c1.get("article_number") and c2.get("article_number")) else 0.0
        is_non_article_pair = 1.0 if (not c1.get("article_number") and not c2.get("article_number")) else 0.0
    else:
        is_same_doc = 0.0
        is_article_pair = 1.0 if c1.get("article_number") else 0.0
        is_non_article_pair = 1.0 if not c1.get("article_number") else 0.0

    # Family 3: Rank Profile
    ranks = [float(c.get("rank", 99.0)) for c in candidate_items]
    min_rank = min(ranks)
    max_rank = max(ranks)
    mean_rank = sum(ranks) / len(ranks)
    max_recip_rank = 1.0 / min_rank if min_rank > 0 else 0.0
    min_recip_rank = 1.0 / max_rank if max_rank > 0 else 0.0
    is_top1 = 1.0 if min_rank == 1.0 else 0.0
    is_top3 = 1.0 if min_rank <= 3.0 else 0.0
    is_top5 = 1.0 if min_rank <= 5.0 else 0.0
    is_top10 = 1.0 if min_rank <= 10.0 else 0.0
    appended_min_rank = ranks[1] if cardinality == 2 else 0.0

    scores = [float(c.get("score", 0.0)) for c in candidate_items]
    max_orig_score = max(scores)
    min_orig_score = min(scores)

    # Family 4: Query-Candidate Text & Lexical Alignment
    q_norm = unicodedata.normalize("NFC", question.lower())
    q_tokens = [t.strip(",.:;\"'()[]{}?!`") for t in q_norm.split() if t.strip(",.:;\"'()[]{}?!`")]
    q_char_len = float(len(question))
    q_word_count = float(len(q_tokens))

    c_norm = unicodedata.normalize("NFC", rendered_text.lower())
    c_tokens = [t.strip(",.:;\"'()[]{}?!`") for t in c_norm.split() if t.strip(",.:;\"'()[]{}?!`")]
    cand_char_len = float(len(rendered_text))
    cand_word_count = float(len(c_tokens))

    char_len_ratio = cand_char_len / (q_char_len + 1.0)
    word_len_ratio = cand_word_count / (q_word_count + 1.0)

    q_set = set(q_tokens)
    c_set = set(c_tokens)
    overlap = q_set.intersection(c_set)
    token_overlap_count = float(len(overlap))
    token_overlap_ratio = token_overlap_count / float(len(q_set)) if q_set else 0.0

    q_numbers = set(NUMBER_REGEX.findall(question))
    c_numbers = set(NUMBER_REGEX.findall(rendered_text))
    number_overlap_count = float(len(q_numbers.intersection(c_numbers)))

    art_mentioned = 1.0 if "điều" in q_set else 0.0
    legal_kw_in_cand = [t for t in c_tokens if t in LEGAL_KEYWORDS]
    legal_kw_count = float(len(legal_kw_in_cand))
    legal_kw_density = legal_kw_count / (cand_word_count + 1.0)
    has_dieu_heading = 1.0 if "điều" in c_set else 0.0

    # Family 5: Redundancy & Incumbent Overlap
    if cardinality == 2 and c2 is not None:
        t1_set = set(c1.get("text", "").lower().split())
        t2_set = set(c2.get("text", "").lower().split())
        union = t1_set.union(t2_set)
        pair_jaccard = float(len(t1_set.intersection(t2_set))) / float(len(union)) if union else 0.0
        pair_len_diff = float(abs(len(c1.get("text", "")) - len(c2.get("text", ""))))
    else:
        pair_jaccard = 0.0
        pair_len_diff = 0.0

    if incumbent_text:
        inc_tokens = set(incumbent_text.lower().split())
        inc_overlap = float(len(c_set.intersection(inc_tokens))) / float(len(inc_tokens)) if inc_tokens else 0.0
    else:
        inc_overlap = 1.0 if has_incumbent else 0.0

    features = [
        float(cardinality), has_incumbent, has_wider, has_non_article, is_hybrid_wider_na,
        num_distinct_branches, is_same_doc, is_article_pair, is_non_article_pair,
        min_rank, max_rank, mean_rank, max_recip_rank, min_recip_rank,
        is_top1, is_top3, is_top5, is_top10, appended_min_rank, max_orig_score, min_orig_score,
        q_char_len, q_word_count, cand_char_len, cand_word_count, char_len_ratio, word_len_ratio,
        token_overlap_count, token_overlap_ratio, number_overlap_count, art_mentioned,
        legal_kw_count, legal_kw_density, has_dieu_heading, pair_jaccard, pair_len_diff, inc_overlap,
    ]
    assert len(features) == 37, f"Extracted {len(features)} features, expected 37."
    return features
