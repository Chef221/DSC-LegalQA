from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

"""Fail-closed artifact integrity verification."""

import hashlib
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("verify_artifacts")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    manifest_file = root / "artifacts" / "MANIFEST.json"

    if not manifest_file.is_file():
        _LOGGER.error(f"FAIL CLOSED: Manifest file not found at {manifest_file}")
        sys.exit(1)

    with open(manifest_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    _LOGGER.info(f"Verifying {len(entries)} production artifacts against manifest...")
    failures = 0

    for item in entries:
        rel_path = item["path"]
        expected_sha = item["sha256"]
        expected_bytes = item.get("bytes")
        target_file = root / rel_path

        if not target_file.is_file():
            _LOGGER.error(f"[FAIL] Missing file: {rel_path}")
            failures += 1
            continue

        actual_bytes = target_file.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            _LOGGER.error(f"[FAIL] Byte count mismatch for {rel_path}: actual {actual_bytes} != expected {expected_bytes}")
            failures += 1
            continue

        actual_sha = compute_sha256(target_file)
        if actual_sha != expected_sha:
            _LOGGER.error(f"[FAIL] SHA256 mismatch for {rel_path}: actual {actual_sha} != expected {expected_sha}")
            failures += 1
            continue

        _LOGGER.info(f"[PASS] {rel_path} ({actual_bytes:,} bytes, SHA256: {actual_sha[:16]}...)")

    if failures > 0:
        _LOGGER.error(f"FAIL CLOSED: {failures} artifact(s) failed integrity check.")
        sys.exit(1)

    _LOGGER.info("[SUCCESS] All frozen production artifacts verified with 100% integrity.")


if __name__ == "__main__":
    main()
