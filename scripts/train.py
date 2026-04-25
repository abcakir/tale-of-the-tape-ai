from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.features.build_matchup_features import FEATURE_COLUMNS, save_normalized_profiles

DATA_FIGHTS = Path("data/Fights.csv")
DATA_PROFILES = Path("data/Fighters Stats.csv")


def _require_input_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_FIGHTS.exists() or not DATA_PROFILES.exists():
        raise FileNotFoundError(
            "Missing Kaggle input files. Expected: data/Fights.csv and data/Fighters Stats.csv"
        )
    return pd.read_csv(DATA_FIGHTS), pd.read_csv(DATA_PROFILES)


def _result_to_target(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.upper().str.strip()
    mapped = normalized.map({"W": 1, "L": 0})
    return mapped


def build_training_frame(fights: pd.DataFrame, raw_profiles: pd.DataFrame) -> pd.DataFrame:
    profiles_path = save_normalized_profiles(raw_profiles)
    profiles = pd.read_csv(profiles_path)

    if "Fighter_1" not in fights.columns or "Fighter_2" not in fights.columns or "Result_1" not in fights.columns:
        raise ValueError("Fights.csv must contain Fighter_1, Fighter_2, Result_1 columns")

    fights = fights.copy()
    fights["target"] = _result_to_target(fights["Result_1"])
    fights = fights.dropna(subset=["target", "Fighter_1", "Fighter_2"])

    merged = fights.merge(
        profiles.add_prefix("a_"), left_on="Fighter_1", right_on="a_name", how="inner"
    ).merge(
        profiles.add_prefix("b_"), left_on="Fighter_2", right_on="b_name", how="inner"
    )

    df = pd.DataFrame(
        {
            "diff_slpm": merged["a_slpm"] - merged["b_slpm"],
            "diff_sapm": merged["a_sapm"] - merged["b_sapm"],
            "diff_td_avg": merged["a_td_avg"] - merged["b_td_avg"],
            "diff_td_acc": merged["a_td_acc"] - merged["b_td_acc"],
            "diff_td_def": merged["a_td_def"] - merged["b_td_def"],
            "diff_sub_avg": merged["a_sub_avg"] - merged["b_sub_avg"],
            "diff_age": merged["a_age"] - merged["b_age"],
            "diff_reach": merged["a_reach"] - merged["b_reach"],
            "diff_height": merged["a_height"] - merged["b_height"],
            "target": merged["target"].astype(int),
        }
    )

    return df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)


def train_model(df: pd.DataFrame) -> LogisticRegression:
    if len(df) < 100:
        raise ValueError(f"Too few training rows after join/cleaning: {len(df)}")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(train_df[FEATURE_COLUMNS].values, train_df["target"].values)

    p = model.predict_proba(test_df[FEATURE_COLUMNS].values)[:, 1]
    metrics = {
        "log_loss": log_loss(test_df["target"].values, p),
        "brier": brier_score_loss(test_df["target"].values, p),
        "roc_auc": roc_auc_score(test_df["target"].values, p),
    }
    print("[train] Metrics:", {k: round(v, 4) for k, v in metrics.items()})
    return model


def main() -> None:
    print("[train] Loading Kaggle UFC data...")
    fights, profiles = _require_input_data()

    print("[train] Building training frame...")
    df = build_training_frame(fights, profiles)
    out_data = Path("data/processed")
    out_data.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_data / "matchups.csv", index=False)
    print(f"[train] Training rows after cleaning: {len(df)}")

    model = train_model(df)

    out_models = Path("artifacts/models")
    out_models.mkdir(parents=True, exist_ok=True)
    path = out_models / "logreg.joblib"
    joblib.dump(model, path)
    print(f"[train] Saved model: {path}")
    print("[train] Saved fighter profiles: artifacts/fighter_profiles.csv")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "42")
    main()
