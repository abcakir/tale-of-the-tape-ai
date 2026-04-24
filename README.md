# tale-of-the-tape-ai

MVP für UFC-Fight-Predictions mit **FastAPI** und Main-Card-Fokus.

## Features (MVP)
- `/predict?fighter_1=...&fighter_2=...` für Einzel-Matchup.
- `/predict/upcoming-numbered-event` für kommendes nummeriertes UFC-Event (Main Card only).
- Robuster Event-Ingestion-Fallback: Falls keine externe Quelle verfügbar ist, wird ein Fixture-Event genutzt.
- Baseline-Training mit Logistic Regression via `scripts/train.py`.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/train.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Optional: externe Eventquelle
Die API kann ein JSON-Event über `UFC_EVENTS_URL` laden. Erwartetes Schema:

```json
{
  "event_name": "UFC 315",
  "event_number": 315,
  "event_date": "2026-06-13",
  "location": "Las Vegas, NV, USA",
  "bouts": [
    {
      "fighter_a": "A",
      "fighter_b": "B",
      "weight_class": "Lightweight",
      "is_main_card": true
    }
  ]
}
```

## Hinweis
Die derzeitigen Fighter-Features in `src/features/build_matchup_features.py` sind deterministische MVP-Platzhalter. In v1 sollen sie durch Kaggle- und Live-Metriken ersetzt werden.
