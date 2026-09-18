"""Legal text preservation and Unicode NFC normalization."""

import html
import re
import unicodedata

# Regex to detect Vietnamese legal structure headings
ARTICLE_HEADER_PATTERN = re.compile(r"^\s*Điều\s+(\d+[a-zA-Z]?)\.?\s*(.*?)$", re.IGNORECASE | re.MULTILINE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ 	]+")


def normalize_legal_text(text: str) -> str:
    """Normalize text into canonical Unicode NFC while preserving legal entities.

    Invariants preserved:
    - Vietnamese diacritics & tone marks preserved.
    - Article/clause numbers, monetary amounts, percentages, and deadlines preserved.
    - Legal keywords ('không', 'trừ', 'hết hiệu lực', etc.) untouched.
    """
    if not text:
        return ""
    # 1. Canonical Unicode NFC
    norm = unicodedata.normalize("NFC", text)
    # 2. Normalize CRLF to LF
    norm = norm.replace("
", "
").replace("", "
")
    # 3. Strip trailing line spaces
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in norm.split("
")]
    return "
".join(lines).strip()


def clean_html_passage(raw_passage: str) -> str:
    """Clean raw HTML passage while preserving text and legal boundaries."""
    if not raw_passage:
        return ""
    # Unescape HTML entities
    unescaped = html.unescape(raw_passage)
    # Replace common break tags with newlines
    cleaned = re.sub(r"<(?:br|p|div|tr|h\d)[^>]*>", "
", unescaped, flags=re.IGNORECASE)
    cleaned = HTML_TAG_PATTERN.sub("", cleaned)
    return normalize_legal_text(cleaned)


def extract_legal_article_number(text: str) -> str:
    """Extract article number e.g. '12' or '34a' if present in heading."""
    match = ARTICLE_HEADER_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return ""
