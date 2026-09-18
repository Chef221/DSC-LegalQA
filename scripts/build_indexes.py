"""Build BM25 index and dense vector embeddings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import sys
from pathlib import Path

import argparse
import json
from pathlib import Path
import numpy as np
from dsc_legalqa.data.schema import LegalChunk
from dsc_legalqa.retrieval.dense import DenseRetriever


def load_chunks(path: str) -> list[LegalChunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                chunks.append(
                    LegalChunk(
                        chunk_id=d["chunk_id"],
                        document_id=d["document_id"],
                        article_number=d.get("article_number", ""),
                        clause_number=d.get("clause_number", ""),
                        text=d["text"],
                        search_text=d.get("search_text", d["text"]),
                        token_count=d.get("token_count", 0),
                    )
                )
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and Dense index.")
    parser.add_argument("--chunks_path", type=str, default="data/chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/indexes", help="Output directory for index artifacts.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dense index build
    print("Building dense embeddings using Qwen3-Embedding-0.6B...")
    dense = DenseRetriever()
    dense.index_chunks(chunks)
    np.save(out_dir / "dense_embeddings.npy", dense.embeddings)
    print(f"Dense embeddings saved to {out_dir / 'dense_embeddings.npy'}")


if __name__ == "__main__":
    main()
