"""Test Codabench submission format and safety invariants."""

import json
from pathlib import Path


def test_no_raw_competition_data_in_git():
    # Fail-closed check: Ensure no official raw datasets are accidentally present in tracked paths
    forbidden_files = [
        Path("data/train.json"),
        Path("data/warmup.json"),
        Path("data/public-official.json"),
        Path("data/private-official.json"),
        Path("data/selected-contexts.zip"),
        Path("submission.json"),
    ]
    for p in forbidden_files:
        assert not p.exists(), f"FORBIDDEN OFFICIAL DATA LEAKAGE: {p} must not exist in clean repository!"


def test_example_submission_schema():
    example_path = Path("examples/submission.example.json")
    assert example_path.exists()
    with open(example_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    for qid, val in data.items():
        assert isinstance(val, dict)
        assert "answer" in val
        assert isinstance(val["answer"], str)
        assert len(val["answer"].strip()) > 0
