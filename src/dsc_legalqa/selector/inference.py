"""P63 selector deployment inference with verified tau=0.100."""

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from dsc_legalqa.selector.features import extract_37_features, FEATURE_NAMES

FROZEN_MODEL_SHA256 = "1654bf0184f6be7f2bc8a715e4eba5f6d47ebc280f10481fac7a1b09bc424c38"
FROZEN_TAU = 0.100
FROZEN_FEATURES = 37


@dataclass(frozen=True, slots=True)
class P63Decision:
    action: str  # "INCUMBENT" or "OVERRIDE"
    selected_chunk_ids: tuple[str, ...]
    selected_text: str
    predicted_delta: float
    tau_applied: float
    override_applied: bool


class P63Selector:
    """Production P63 selector enforcing frozen SHA256 and tau=0.100."""

    def __init__(
        self,
        model_path: Path | str = "artifacts/p63_selector/P63_DEPLOYMENT_MODEL.pkl",
        meta_path: Path | str = "artifacts/p63_selector/P63_DEPLOYMENT_MODEL_METADATA.json",
    ):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.model: HistGradientBoostingRegressor | None = None
        self.tau = FROZEN_TAU

    def load_and_verify(self):
        import hashlib
        h = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if h != FROZEN_MODEL_SHA256:
            raise RuntimeError(f"P63 Model SHA256 mismatch: {h} != {FROZEN_MODEL_SHA256}")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def select(
        self,
        question: str,
        incumbent_chunk_ids: tuple[str, ...],
        incumbent_text: str,
        override_candidates: list[dict[str, Any]],
    ) -> P63Decision:
        """Select incumbent or override action based on predicted METEOR delta."""
        if self.model is None:
            self.load_and_verify()

        if not override_candidates:
            return P63Decision(
                action="INCUMBENT",
                selected_chunk_ids=incumbent_chunk_ids,
                selected_text=incumbent_text,
                predicted_delta=0.0,
                tau_applied=self.tau,
                override_applied=False,
            )

        feature_rows = []
        for cand in override_candidates:
            feats = extract_37_features(
                question=question,
                candidate_items=cand.get("items", []),
                rendered_text=cand.get("text", ""),
                incumbent_text=incumbent_text,
            )
            feature_rows.append(feats)

        feat_arr = np.array(feature_rows, dtype=np.float32)
        preds = self.model.predict(feat_arr)
        best_idx = int(np.argmax(preds))
        best_pred = float(preds[best_idx])

        if best_pred >= self.tau:
            chosen = override_candidates[best_idx]
            return P63Decision(
                action="OVERRIDE",
                selected_chunk_ids=tuple(chosen.get("chunk_ids", ())),
                selected_text=chosen.get("text", ""),
                predicted_delta=best_pred,
                tau_applied=self.tau,
                override_applied=True,
            )
        else:
            return P63Decision(
                action="INCUMBENT",
                selected_chunk_ids=incumbent_chunk_ids,
                selected_text=incumbent_text,
                predicted_delta=best_pred,
                tau_applied=self.tau,
                override_applied=False,
            )
