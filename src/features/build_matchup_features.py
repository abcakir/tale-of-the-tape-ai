from __future__ import annotations

import hashlib
from typing import Dict


def _stable_score(name: str, offset: int) -> float:
    digest = hashlib.sha256(f"{name}:{offset}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return (value % 1000) / 1000.0


def fighter_profile(name: str) -> Dict[str, float]:
    """Deterministic pseudo-profile for MVP inference without external fighter DB.

    Replace this with real fighter metrics from Kaggle + live updates in v1.
    """
    return {
        "slpm": 2.0 + 4.0 * _stable_score(name, 1),
        "sapm": 2.0 + 4.0 * _stable_score(name, 2),
        "td_avg": 0.2 + 4.0 * _stable_score(name, 3),
        "td_def": 0.3 + 0.7 * _stable_score(name, 4),
        "reach": 65 + 20 * _stable_score(name, 5),
        "age": 23 + 15 * _stable_score(name, 6),
    }


def matchup_features(fighter_a: str, fighter_b: str) -> Dict[str, float]:
    a = fighter_profile(fighter_a)
    b = fighter_profile(fighter_b)
    return {
        "diff_slpm": a["slpm"] - b["slpm"],
        "diff_sapm": a["sapm"] - b["sapm"],
        "diff_td_avg": a["td_avg"] - b["td_avg"],
        "diff_td_def": a["td_def"] - b["td_def"],
        "diff_reach": a["reach"] - b["reach"],
        "diff_age": a["age"] - b["age"],
    }
