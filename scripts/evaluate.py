"""Evaluate submission predictions against gold references using official competition scorer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import sys
from pathlib import Path

import argparse
import json
from dsc_legalqa.evaluation.scorer import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate LegalQA answers.")
    parser.add_argument("--references", type=str, required=True, help="Path to reference answers JSON.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON.")
    args = parser.parse_args()

    with open(args.references, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    with open(args.predictions, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    ref_map = {qid: v["answer"] if isinstance(v, dict) else str(v) for qid, v in ref_data.items()}
    pred_map = {qid: v["answer"] if isinstance(v, dict) else str(v) for qid, v in pred_data.items()}

    results = evaluate_predictions(ref_map, pred_map)
    print("=== Evaluation Results ===")
    print(f"Sample Count: {results['sample_count']}")
    print(f"METEOR (Primary): {results['meteor']:.9f}")
    print(f"ROUGE-L (Secondary): {results['rouge_l']:.9f}")


if __name__ == "__main__":
    main()
