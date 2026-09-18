"""Dataset loading utilities for official competition files."""

import json
from pathlib import Path
from typing import Any
from dsc_legalqa.data.schema import LegalDocument, LegalChunk


def load_official_contexts(contexts_dir_or_zip: Path | str) -> list[LegalDocument]:
    """Load official context_*.json files from a directory."""
    path = Path(contexts_dir_or_zip)
    documents: list[LegalDocument] = []

    if path.is_dir():
        for f in sorted(path.glob("context_*.json")):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                documents.append(
                    LegalDocument(
                        id=str(data.get("id", "")),
                        name=data.get("name", ""),
                        link=data.get("link", ""),
                        passage=data.get("passage", ""),
                    )
                )
    return documents


def load_qa_records(json_path: Path | str) -> dict[str, dict[str, Any]]:
    """Load question-answer mapping from train.json, warmup.json, or public-official.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_legal_chunks(chunks_path: Path | str) -> list[LegalChunk]:
    """Load indexed LegalChunk objects from a JSON or JSONL file."""
    path = Path(chunks_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legal chunks file not found at: {chunks_path}")

    chunks: list[LegalChunk] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    chunks.append(LegalChunk(**item))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    chunks.append(LegalChunk(**item))
            elif isinstance(data, dict):
                for item in data.values():
                    chunks.append(LegalChunk(**item))

    if not chunks:
        raise ValueError(f"No valid LegalChunk records parsed from {chunks_path}")
    return chunks
