# tale-of-the-tape-ai

FastAPI-basierter UFC-Predictor (Main Card only) mit Kaggle-Trainingspipeline.

## Was ist neu
- Echte Trainingspipeline auf Basis von `data/Fights.csv` und `data/Fighters Stats.csv`.
- Keine pseudo-zufälligen Feature-Platzhalter mehr.
- API gibt klare `503`-Fehler zurück, wenn Modell/Profile fehlen.

## Voraussetzungen
Lege folgende Dateien lokal ab:
- `data/Fights.csv`
- `data/Fighters Stats.csv`

## Training
```bash
pip install -r requirements.txt
python scripts/train.py
```

Training erzeugt:
- `artifacts/models/logreg.joblib`
- `artifacts/fighter_profiles.csv`
- `data/processed/matchups.csv`

## API starten
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints
- `GET /` → Health + `predictor_ready`
- `GET /predict?fighter_1=...&fighter_2=...`
- `GET /predict/upcoming-numbered-event?main_card_only=true`

## Optional: externe Eventquelle
Setze `UFC_EVENTS_URL` auf eine JSON-Quelle mit folgendem Schema:

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
