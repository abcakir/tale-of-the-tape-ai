from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.features.build_matchup_features import FEATURE_COLUMNS, load_profiles, matchup_features


class PredictorNotReadyError(RuntimeError):
    pass


class Predictor:
    def __init__(
        self,
        model_path: str = "artifacts/models/logreg.joblib",
        profiles_path: str = "artifacts/fighter_profiles.csv",
    ) -> None:
        self.model_path = Path(model_path)
        self.profiles_path = Path(profiles_path)

        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
        self.profiles: pd.DataFrame | None = load_profiles(str(self.profiles_path)) if self.profiles_path.exists() else None

    def ready(self) -> bool:
        return self.model is not None and self.profiles is not None

    def predict_matchup(self, fighter_a: str, fighter_b: str) -> tuple[float, Dict[str, float]]:
        if not self.ready():
            raise PredictorNotReadyError(
                "Model or fighter profiles missing. Run `python scripts/train.py` with Kaggle CSVs in data/."
            )

        assert self.profiles is not None
        features = matchup_features(fighter_a, fighter_b, profiles=self.profiles)
        x = np.array([[features[k] for k in FEATURE_COLUMNS]])
        p = float(self.model.predict_proba(x)[0, 1])
        return p, features


def top_key_factors(features: Dict[str, float], n: int = 3) -> List[str]:
    ranked = sorted(features.items(), key=lambda kv: abs(kv[1]), reverse=True)
    reasons = []
    for name, value in ranked[:n]:
        direction = "Vorteil Fighter A" if value > 0 else "Vorteil Fighter B"
        reasons.append(f"{name}: {value:.2f} ({direction})")
    return reasons
