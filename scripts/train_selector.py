from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

"""Train P63 HistGradientBoostingRegressor selector on 37 reference-free features.

Fail-Closed Enforcement:
- No synthetic random fallbacks.
- Strictly validates 37 features schema.
- Uses validated production hyperparameters matching P63 deployment authority.
"""

import argparse
import hashlib
import json
import logging
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from dsc_legalqa.selector.features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("train_selector")

DEFAULT_HYPERPARAMS = {
    "max_iter": 100,
    "max_depth": 6,
    "learning_rate": 0.08,
    "min_samples_leaf": 20,
    "random_state": 42,
}

EXPECTED_FEATURE_COUNT = 37
EXPECTED_TRAIN_CANDIDATE_ROWS = 215147
EXPECTED_TRAIN_QID_COUNT = 5300
FROZEN_TAU_DEPLOY = 0.100


def main():
    parser = argparse.ArgumentParser(description="Train P63 Evidence Selector.")
    parser.add_argument("--features_path", type=str, default="data/p63_training_features.npz", help="Path to pre-extracted P63 37-feature npz.")
    parser.add_argument("--output_dir", type=str, default="artifacts/p63_selector", help="Output directory for trained model & metadata.")
    args = parser.parse_args()

    feat_path = Path(args.features_path)
    if not feat_path.is_file():
        raise FileNotFoundError(
            f"FAIL CLOSED: Required P63 training feature artifact not found at: {feat_path}\n"
            f"Random synthetic fallback is prohibited. To reproduce P63 training, provide the authentic "
            f"training features extracted from official training data (approx {EXPECTED_TRAIN_CANDIDATE_ROWS:,} candidate rows "
            f"across {EXPECTED_TRAIN_QID_COUNT} QIDs)."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info(f"Loading training features from {feat_path}...")
    data = np.load(feat_path)
    X, y = data["X"], data["y"]

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, got {X.shape[1]}")

    _LOGGER.info(f"Training HistGradientBoostingRegressor on {X.shape[0]} rows with {X.shape[1]} features...")
    _LOGGER.info(f"Hyperparameters: {DEFAULT_HYPERPARAMS}")

    model = HistGradientBoostingRegressor(**DEFAULT_HYPERPARAMS)
    model.fit(X, y)

    model_file = out_dir / "P63_RETRAINED_MODEL.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Compute model hash
    model_bytes = model_file.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()

    meta = {
        "model_file": model_file.name,
        "model_sha256": model_sha256,
        "model_family": "HistGradientBoostingRegressor",
        "hyperparameters": DEFAULT_HYPERPARAMS,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "tau_deploy": FROZEN_TAU_DEPLOY,
        "total_training_rows": int(X.shape[0]),
    }
    meta_file = out_dir / "P63_RETRAINED_MODEL_METADATA.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _LOGGER.info(f"[SUCCESS] Model saved to {model_file} (SHA256: {model_sha256})")
    _LOGGER.info(f"[SUCCESS] Metadata saved to {meta_file}")


if __name__ == "__main__":
    main()
