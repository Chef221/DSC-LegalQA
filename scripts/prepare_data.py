"""Prepare legal documents into normalized retrieval chunks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import sys
from pathlib import Path

import argparse
import json
import re
from pathlib import Path
from dsc_legalqa.data.loader import load_official_contexts
from dsc_legalqa.preprocessing.normalizer import clean_html_passage, extract_legal_article_number
from dsc_legalqa.data.schema import LegalChunk

ARTICLE_REGEX = re.compile(r"(?=(?:^|\n)\s*Điều\s+(\d+[a-zA-Z]?)\.?)", re.IGNORECASE)


def chunk_document(doc_id: str | int, text: str) -> list[LegalChunk]:
    """Segment document text into article-level chunks (1 Điều = 1 chunk)."""
    clean_text = clean_html_passage(text)
    if not clean_text:
        return []

    # Find article boundaries
    splits = ARTICLE_REGEX.split(clean_text)
    chunks: list[LegalChunk] = []

    if len(splits) <= 1:
        # Fallback chunk if no explicit 'Điều' header
        chunks.append(
            LegalChunk(
                chunk_id=f"{doc_id}__fallback__rc0",
                document_id=str(doc_id),
                article_number="",
                clause_number="",
                text=clean_text,
                search_text=clean_text.lower(),
                token_count=len(clean_text.split()),
            )
        )
        return chunks

    # Process article parts
    # splits[0] is preamble before Điều 1
    if splits[0].strip():
        chunks.append(
            LegalChunk(
                chunk_id=f"{doc_id}__preamble__rc0",
                document_id=str(doc_id),
                article_number="",
                clause_number="",
                text=splits[0].strip(),
                search_text=splits[0].strip().lower(),
                token_count=len(splits[0].strip().split()),
            )
        )

    i = 1
    while i < len(splits):
        art_num = splits[i].strip().lower()
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        art_text = f"Điều {art_num}. {content}".strip()
        chunks.append(
            LegalChunk(
                chunk_id=f"{doc_id}__art_{art_num}__rc0",
                document_id=str(doc_id),
                article_number=art_num,
                clause_number="",
                text=art_text,
                search_text=art_text.lower(),
                token_count=len(art_text.split()),
            )
        )
        i += 2

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Prepare legal corpus into chunks.")
    parser.add_argument("--data_dir", type=str, default="data/selected-contexts", help="Path to context files.")
    parser.add_argument("--output_path", type=str, default="data/chunks.jsonl", help="Output path for chunks.")
    args = parser.parse_args()

    print(f"Loading contexts from {args.data_dir}...")
    docs = load_official_contexts(args.data_dir)
    print(f"Loaded {len(docs)} documents.")

    all_chunks: list[LegalChunk] = []
    for doc in docs:
        c_list = chunk_document(doc.id, doc.passage)
        all_chunks.extend(c_list)

    print(f"Produced {len(all_chunks)} legal chunks.")
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            row = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "article_number": c.article_number,
                "clause_number": c.clause_number,
                "text": c.text,
                "search_text": c.search_text,
                "token_count": c.token_count,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved chunks to {out}")


if __name__ == "__main__":
    main()
