from fastapi import FastAPI

app = FastAPI(title="TaleOfTheTape API")

@app.get("/")
def health_check():
    return {"status": "OK", "message": "API is running!"}

@app.get("/predict")
def predict_matchup(fighter_1: str, fighter_2: str):
    return{
        "matchup": f"{fighter_1} vs. {fighter_2}",
        "win_probability_fighter_1": 55.1,
        "win_probability_fighter_2": 44.9,
        "status": "just a placeholder"
    }
