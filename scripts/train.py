from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.model.predictor import FEATURE_ORDER


def load_training_frame() -> pd.DataFrame:
    """Loads engineered features.

    For MVP we support either:
    1) data/processed/matchups.csv with FEATURE_ORDER + target
    2) synthetic fallback dataset for local smoke tests
    """
    path = Path("data/processed/matchups.csv")
    if path.exists():
        df = pd.read_csv(path)
        required = set(FEATURE_ORDER + ["target"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return df

    rng = np.random.default_rng(42)
    n = 1500
    X = rng.normal(size=(n, len(FEATURE_ORDER)))
    w = np.array([0.45, -0.20, 0.30, 0.25, 0.03, -0.04])
    probs = 1 / (1 + np.exp(-(X @ w)))
    y = (rng.uniform(size=n) < probs).astype(int)
    df = pd.DataFrame(X, columns=FEATURE_ORDER)
    df["target"] = y
    return df


def main() -> None:
    print("[train] Loading training frame...")
    df = load_training_frame()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_ORDER].values
    y_train = train_df["target"].values
    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df["target"].values

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    metrics = {
        "log_loss": log_loss(y_test, p),
        "brier": brier_score_loss(y_test, p),
        "roc_auc": roc_auc_score(y_test, p),
    }
    print("[train] Metrics:", {k: round(v, 4) for k, v in metrics.items()})

    out_dir = Path("artifacts/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "logreg.joblib"
    joblib.dump(model, model_path)
    print(f"[train] Saved model: {model_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "42")
    main()
