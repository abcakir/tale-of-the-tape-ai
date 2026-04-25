from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List


@dataclass(frozen=True)
class FightCardBout:
    fighter_a: str
    fighter_b: str
    weight_class: str
    is_main_card: bool = True


@dataclass(frozen=True)
class UFCEvent:
    event_name: str
    event_number: int
    event_date: date
    location: str
    bouts: List[FightCardBout]
    source: str


def fallback_upcoming_numbered_event() -> UFCEvent:
    """Fallback event fixture for local development.

    Used if upstream internet source is unavailable.
    """
    return UFCEvent(
        event_name="UFC 315",
        event_number=315,
        event_date=date(2026, 6, 13),
        location="Las Vegas, NV, USA",
        source="fallback-fixture",
        bouts=[
            FightCardBout("Fighter Red A", "Fighter Blue A", "Heavyweight", True),
            FightCardBout("Fighter Red B", "Fighter Blue B", "Welterweight", True),
            FightCardBout("Fighter Red C", "Fighter Blue C", "Lightweight", True),
            FightCardBout("Fighter Red D", "Fighter Blue D", "Middleweight", True),
            FightCardBout("Fighter Red E", "Fighter Blue E", "Bantamweight", True),
        ],
    )
