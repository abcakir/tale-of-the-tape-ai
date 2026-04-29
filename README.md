# tale-of-the-tape-ai

FastAPI-basierter UFC-Predictor (Main Card only) mit Kaggle-Trainingspipeline.

## Schnellstart: Kaggle-Daten in `data/`
Wenn du das Dataset `aminealibi/ufc-fights-fighters-and-events-dataset` direkt holen willst:

```bash
bash scripts/fetch_kaggle_data.sh
```

Der Download landet bewusst in `data/` (nicht in `data/raw`).

> Voraussetzung lokal: Kaggle CLI + API-Key (`~/.kaggle/kaggle.json`).

## Was ist neu
- Echte Trainingspipeline auf Basis von Fight- und Fighter-Stats CSVs.
- Keine pseudo-zufälligen Feature-Platzhalter mehr.
- API gibt klare `503`-Fehler zurück, wenn Modell/Profile fehlen.
- CSV-Pfade sind flexibel über Umgebungsvariablen konfigurierbar.
- Robustere Spalten-Erkennung für Fight-CSV (`Fighter_1`/`RedFighter` etc., `Result_1`/`Winner` etc.).

## Dateneingang
Standardmäßig sucht das Training folgende Dateinamen:
- Fights: `data/Fights.csv`, `data/fights.csv`, `data/raw/Fights.csv`, `data/raw/fights.csv`, `data/ufc_fights.csv`
- Fighter Stats: `data/Fighters Stats.csv`, `data/fighter_stats.csv`, `data/Fighters_Stats.csv`, `data/raw/Fighters Stats.csv`, `data/raw/fighter_stats.csv`, `data/ufc_fighters.csv`

Optional kannst du explizite Pfade setzen:
- `FIGHTS_CSV=/abs/or/rel/path/to/fights.csv`
- `FIGHTER_STATS_CSV=/abs/or/rel/path/to/fighter_stats.csv`

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
