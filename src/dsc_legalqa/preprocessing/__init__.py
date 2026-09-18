"""Preprocessing and legal text normalization."""

from dsc_legalqa.preprocessing.normalizer import (
    normalize_legal_text,
    clean_html_passage,
    extract_legal_article_number,
)

__all__ = ["normalize_legal_text", "clean_html_passage", "extract_legal_article_number"]
