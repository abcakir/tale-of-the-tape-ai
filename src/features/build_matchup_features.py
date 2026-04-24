from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

FEATURE_COLUMNS = [
    "diff_slpm",
    "diff_sapm",
    "diff_td_avg",
    "diff_td_acc",
    "diff_td_def",
    "diff_sub_avg",
    "diff_age",
    "diff_reach",
    "diff_height",
]

PROFILE_MAP = {
    "slpm": ["SLpM", "SLPM", "slpm"],
    "sapm": ["SApM", "SAPM", "sapm"],
    "td_avg": ["TD Avg.", "TD Avg", "TD_Avg", "td_avg"],
    "td_acc": ["TD Acc.", "TD Acc", "TD_Acc", "td_acc"],
    "td_def": ["TD Def.", "TD Def", "TD_Def", "td_def"],
    "sub_avg": ["Sub. Avg.", "Sub Avg", "Sub_Avg", "sub_avg"],
    "age": ["Age", "age"],
    "reach": ["Reach", "reach"],
    "height": ["Height", "height"],
}


class UnknownFighterError(ValueError):
    pass


def _pick_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _name_column(columns: Iterable[str]) -> str:
    candidates = ["Name", "Fighter", "fighter_name", "name"]
    col = _pick_col(columns, candidates)
    if col is None:
        raise ValueError("Could not find fighter name column in fighter profile data")
    return col


def normalize_profiles(raw_profiles: pd.DataFrame) -> pd.DataFrame:
    name_col = _name_column(raw_profiles.columns)
    normalized = pd.DataFrame({"name": raw_profiles[name_col].astype(str).str.strip()})

    for output_col, candidates in PROFILE_MAP.items():
        input_col = _pick_col(raw_profiles.columns, candidates)
        if input_col is None:
            normalized[output_col] = 0.0
        else:
            normalized[output_col] = pd.to_numeric(raw_profiles[input_col], errors="coerce").fillna(0.0)

    return normalized.drop_duplicates(subset=["name"]).reset_index(drop=True)


def save_normalized_profiles(raw_profiles: pd.DataFrame, out_path: str = "artifacts/fighter_profiles.csv") -> Path:
    df = normalize_profiles(raw_profiles)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def load_profiles(path: str = "artifacts/fighter_profiles.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fighter profile file not found: {path}")
    df = pd.read_csv(p)
    required = {"name", *PROFILE_MAP.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    return df


def _get_fighter_row(profiles: pd.DataFrame, fighter_name: str) -> pd.Series:
    match = profiles.loc[profiles["name"].str.lower() == fighter_name.strip().lower()]
    if match.empty:
        raise UnknownFighterError(f"Unknown fighter '{fighter_name}'. Retrain with updated fighter profiles.")
    return match.iloc[0]


def _pair_diff(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    return {
        "diff_slpm": float(a["slpm"] - b["slpm"]),
        "diff_sapm": float(a["sapm"] - b["sapm"]),
        "diff_td_avg": float(a["td_avg"] - b["td_avg"]),
        "diff_td_acc": float(a["td_acc"] - b["td_acc"]),
        "diff_td_def": float(a["td_def"] - b["td_def"]),
        "diff_sub_avg": float(a["sub_avg"] - b["sub_avg"]),
        "diff_age": float(a["age"] - b["age"]),
        "diff_reach": float(a["reach"] - b["reach"]),
        "diff_height": float(a["height"] - b["height"]),
    }


def matchup_features(fighter_a: str, fighter_b: str, profiles: pd.DataFrame) -> Dict[str, float]:
    a = _get_fighter_row(profiles, fighter_a)
    b = _get_fighter_row(profiles, fighter_b)
    return _pair_diff(a, b)


def matchup_matrix(
    pairs: Iterable[Tuple[str, str]], profiles: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        row = matchup_features(a, b, profiles)
        row["fighter_a"] = a
        row["fighter_b"] = b
        rows.append(row)
    return pd.DataFrame(rows)
