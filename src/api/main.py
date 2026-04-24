from fastapi import FastAPI, Query

from src.api.schemas import EventPredictionResponse, FightPrediction
from src.features.build_matchup_features import matchup_features
from src.ingestion.fetch_events import fetch_upcoming_numbered_event
from src.model.predictor import Predictor, top_key_factors

app = FastAPI(title="TaleOfTheTape API", version="0.1.0")
predictor = Predictor()


@app.get("/")
def health_check():
    return {"status": "OK", "message": "API is running!"}


@app.get("/predict", response_model=FightPrediction)
def predict_matchup(fighter_1: str, fighter_2: str):
    features = matchup_features(fighter_1, fighter_2)
    p_a = predictor.predict_proba(features)
    return FightPrediction(
        fighter_a=fighter_1,
        fighter_b=fighter_2,
        p_a_win=round(p_a, 4),
        p_b_win=round(1 - p_a, 4),
        confidence=round(abs(p_a - 0.5) * 2, 4),
        key_factors=top_key_factors(features),
    )


@app.get("/predict/upcoming-numbered-event", response_model=EventPredictionResponse)
def predict_upcoming_numbered_event(main_card_only: bool = Query(True)):
    event = fetch_upcoming_numbered_event()
    fights = []
    for bout in event.bouts:
        if main_card_only and not bout.is_main_card:
            continue
        features = matchup_features(bout.fighter_a, bout.fighter_b)
        p_a = predictor.predict_proba(features)
        fights.append(
            FightPrediction(
                fighter_a=bout.fighter_a,
                fighter_b=bout.fighter_b,
                p_a_win=round(p_a, 4),
                p_b_win=round(1 - p_a, 4),
                confidence=round(abs(p_a - 0.5) * 2, 4),
                key_factors=top_key_factors(features),
            )
        )

    return EventPredictionResponse(
        event_name=event.event_name,
        event_number=event.event_number,
        event_date=event.event_date,
        location=event.location,
        fights=fights,
        source=event.source,
    )
