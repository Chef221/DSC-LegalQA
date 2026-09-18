"""Build and validate official submission.zip for Codabench."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def validate_submission_json(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("submission.json root must be a JSON object mapping question_id -> {'answer': text}")
    for qid, val in data.items():
        if not isinstance(val, dict) or "answer" not in val or not isinstance(val["answer"], str):
            raise ValueError(f"Invalid entry for question_id '{qid}': expected {{'answer': string}}, got {val}")


def main():
    parser = argparse.ArgumentParser(description="Package submission.zip for Codabench.")
    parser.add_argument("--input_json", type=str, default="submission.json", help="Path to submission.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip", help="Path to output submission.zip")
    args = parser.parse_args()

    input_p = Path(args.input_json)
    if not input_p.exists():
        raise FileNotFoundError(f"Input file {input_p} not found.")

    with open(input_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_submission_json(data)
    print(f"Validated {len(data)} answer entries in {input_p}.")

    out_p = Path(args.output_zip)
    with zipfile.ZipFile(out_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(input_p, arcname="submission.json")

    # Verify zip content
    with zipfile.ZipFile(out_p, "r") as zf:
        namelist = zf.namelist()
        if namelist != ["submission.json"]:
            raise RuntimeError(f"Submission zip must contain only 'submission.json', found: {namelist}")

    h = hashlib.sha256(out_p.read_bytes()).hexdigest()
    print(f"Successfully packaged {out_p} (SHA256: {h})")


if __name__ == "__main__":
    main()
