"""Dataset loading utilities for official competition files."""

import json
from pathlib import Path
from typing import Any, Generator
from dsc_legalqa.data.schema import LegalDocument


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
                        id=data.get("id", ""),
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
