"""Test frozen artifact provenance against strict cryptographic hashes."""

import hashlib
import json
from pathlib import Path

FROZEN_EXPECTATIONS = {
    "artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl": "1654bf0184f6be7f2bc8a715e4eba5f6d47ebc280f10481fac7a1b09bc424c38",
    "artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json": "ef59ed7efe1a47edf76b417dd53660eb15bfb52fc2a7bc314bb3dd58ca0e27ac",
    "artifacts/p70_adapter/adapter_model.safetensors": "193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c",
    "artifacts/p70_adapter/adapter_config.json": "2761928c3f17aa894bd2e7793db0d00a2e7c1b0821a7d864dc7bf411dc603f37",
    "src/dsc_legalqa/postprocess/token_suffix_loop_sanitizer.py": "1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_artifacts_exist_and_match_sha256():
    repo_root = Path(__file__).resolve().parent.parent
    for rel_path, expected_sha in FROZEN_EXPECTATIONS.items():
        full_path = repo_root / rel_path
        assert full_path.is_file(), f"Missing artifact: {rel_path}"
        actual_sha = compute_sha256(full_path)
        assert actual_sha == expected_sha, f"SHA256 mismatch for {rel_path}: {actual_sha} != {expected_sha}"


def test_manifest_json_matches_artifacts():
    repo_root = Path(__file__).resolve().parent.parent
    manifest_file = repo_root / "artifacts" / "MANIFEST.json"
    assert manifest_file.is_file()

    with open(manifest_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    assert len(entries) == 5
    manifest_dict = {e["path"]: e["sha256"] for e in entries}
    for rel_path, expected_sha in FROZEN_EXPECTATIONS.items():
        assert rel_path in manifest_dict
        assert manifest_dict[rel_path] == expected_sha


def test_p63_metadata_content():
    repo_root = Path(__file__).resolve().parent.parent
    meta_file = repo_root / "artifacts" / "p63_selector" / "P63_DEPLOYMENT_MODEL_METADATA.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_family"] == "HistGradientBoostingRegressor"
    assert data["feature_count"] == 37
    assert data["tau_deploy"] == 0.1
    assert data["total_training_candidate_rows"] == 215147
    assert data["training_qid_count"] == 5300
