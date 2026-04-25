from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict

import requests

from .events import FightCardBout, UFCEvent, fallback_upcoming_numbered_event


def _parse_event(payload: Dict[str, Any], source: str) -> UFCEvent:
    bouts = [
        FightCardBout(
            fighter_a=b["fighter_a"],
            fighter_b=b["fighter_b"],
            weight_class=b.get("weight_class", "Unknown"),
            is_main_card=bool(b.get("is_main_card", False)),
        )
        for b in payload.get("bouts", [])
        if b.get("is_main_card", False)
    ]
    return UFCEvent(
        event_name=payload["event_name"],
        event_number=int(payload["event_number"]),
        event_date=date.fromisoformat(payload["event_date"]),
        location=payload.get("location", "TBD"),
        bouts=bouts,
        source=source,
    )


def fetch_upcoming_numbered_event(timeout_seconds: int = 8) -> UFCEvent:
    """Fetches upcoming numbered UFC event (main-card only).

    Expects an optional JSON endpoint via UFC_EVENTS_URL env variable.
    Falls back to a fixture when unavailable.
    """
    url = os.getenv("UFC_EVENTS_URL")
    if not url:
        return fallback_upcoming_numbered_event()

    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return _parse_event(payload, source=url)
    except Exception:
        return fallback_upcoming_numbered_event()
