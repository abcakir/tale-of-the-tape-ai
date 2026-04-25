from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.features.build_matchup_features import FEATURE_COLUMNS, save_normalized_profiles

DEFAULT_FIGHTS_CANDIDATES = [
    "data/Fights.csv",
    "data/fights.csv",
    "data/raw/Fights.csv",
    "data/raw/fights.csv",
    "data/ufc_fights.csv",
]
DEFAULT_STATS_CANDIDATES = [
    "data/Fighters Stats.csv",
    "data/fighter_stats.csv",
    "data/Fighters_Stats.csv",
    "data/raw/Fighters Stats.csv",
    "data/raw/fighter_stats.csv",
    "data/ufc_fighters.csv",
]

FIGHTER_1_CANDIDATES = ["Fighter_1", "RedFighter", "r_fighter", "red_fighter", "fighter_red"]
FIGHTER_2_CANDIDATES = ["Fighter_2", "BlueFighter", "b_fighter", "blue_fighter", "fighter_blue"]
RESULT_CANDIDATES = ["Result_1", "Winner", "winner", "winner_name", "winning_corner"]


def _pick_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _first_existing(candidates: list[str]) -> Path | None:
    for path in candidates:
        p = Path(path)
        if p.exists():
            return p
    return None


def _resolve_input_path(env_key: str, candidates: list[str]) -> Path | None:
    override = os.getenv(env_key)
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"{env_key} points to missing file: {override}")
    return _first_existing(candidates)


def _require_input_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fights_path = _resolve_input_path("FIGHTS_CSV", DEFAULT_FIGHTS_CANDIDATES)
    stats_path = _resolve_input_path("FIGHTER_STATS_CSV", DEFAULT_STATS_CANDIDATES)

    if fights_path is None or stats_path is None:
        raise FileNotFoundError(
            "Missing Kaggle input files. Set FIGHTS_CSV/FIGHTER_STATS_CSV or place files at one of:\n"
            f"- fights: {DEFAULT_FIGHTS_CANDIDATES}\n"
            f"- fighter stats: {DEFAULT_STATS_CANDIDATES}"
        )

    print(f"[train] Using fights CSV: {fights_path}")
    print(f"[train] Using fighter stats CSV: {stats_path}")
    return pd.read_csv(fights_path), pd.read_csv(stats_path)


def _result_to_target(result: pd.Series, fighter_1: pd.Series, fighter_2: pd.Series) -> pd.Series:
    normalized = result.astype(str).str.upper().str.strip()

    # direct W/L format for fighter_1
    direct = normalized.map({"W": 1, "L": 0})

    # red/blue corner format
    corner = normalized.map(
        {
            "RED": 1,
            "R": 1,
            "BLUE": 0,
            "B": 0,
            "FIGHTER_1": 1,
            "FIGHTER_2": 0,
        }
    )

    # winner name format
    f1 = fighter_1.astype(str).str.upper().str.strip()
    f2 = fighter_2.astype(str).str.upper().str.strip()
    winner_name = pd.Series(index=result.index, dtype="float64")
    winner_name[normalized == f1] = 1.0
    winner_name[normalized == f2] = 0.0

    target = direct.fillna(corner).fillna(winner_name)
    return target


def build_training_frame(fights: pd.DataFrame, raw_profiles: pd.DataFrame) -> pd.DataFrame:
    profiles_path = save_normalized_profiles(raw_profiles)
    profiles = pd.read_csv(profiles_path)

    f1_col = _pick_col(fights.columns, FIGHTER_1_CANDIDATES)
    f2_col = _pick_col(fights.columns, FIGHTER_2_CANDIDATES)
    result_col = _pick_col(fights.columns, RESULT_CANDIDATES)
    if not f1_col or not f2_col or not result_col:
        raise ValueError(
            "Could not resolve required fight columns. Need fighter_1/fighter_2/result equivalents. "
            f"Available columns: {list(fights.columns)[:40]}"
        )

    fights = fights.copy()
    fights["fighter_1"] = fights[f1_col].astype(str).str.strip()
    fights["fighter_2"] = fights[f2_col].astype(str).str.strip()
    fights["target"] = _result_to_target(fights[result_col], fights["fighter_1"], fights["fighter_2"])
    fights = fights.dropna(subset=["target", "fighter_1", "fighter_2"])

    merged = fights.merge(
        profiles.add_prefix("a_"), left_on="fighter_1", right_on="a_name", how="inner"
    ).merge(
        profiles.add_prefix("b_"), left_on="fighter_2", right_on="b_name", how="inner"
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
