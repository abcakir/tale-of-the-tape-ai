from datetime import date
from typing import List

from pydantic import BaseModel, Field


class FightPrediction(BaseModel):
    fighter_a: str
    fighter_b: str
    p_a_win: float = Field(ge=0.0, le=1.0)
    p_b_win: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    key_factors: List[str]


class EventPredictionResponse(BaseModel):
    event_name: str
    event_number: int
    event_date: date
    location: str
    fights: List[FightPrediction]
    source: str
