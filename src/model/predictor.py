from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

FEATURE_ORDER = [
    "diff_slpm",
    "diff_sapm",
    "diff_td_avg",
    "diff_td_def",
    "diff_reach",
    "diff_age",
]


class Predictor:
    def __init__(self, model_path: str = "artifacts/models/logreg.joblib") -> None:
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None

    def predict_proba(self, features: Dict[str, float]) -> float:
        x = np.array([[features[k] for k in FEATURE_ORDER]])
        if self.model is not None:
            return float(self.model.predict_proba(x)[0, 1])

        # fallback heuristic (kept transparent for MVP)
        z = (
            0.40 * features["diff_slpm"]
            - 0.25 * features["diff_sapm"]
            + 0.30 * features["diff_td_avg"]
            + 0.20 * features["diff_td_def"]
            + 0.02 * features["diff_reach"]
            - 0.03 * features["diff_age"]
        )
        return float(1.0 / (1.0 + np.exp(-z)))


def top_key_factors(features: Dict[str, float], n: int = 3) -> List[str]:
    ranked = sorted(features.items(), key=lambda kv: abs(kv[1]), reverse=True)
    reasons = []
    for name, value in ranked[:n]:
        direction = "Vorteil Fighter A" if value > 0 else "Vorteil Fighter B"
        reasons.append(f"{name}: {value:.2f} ({direction})")
    return reasons
