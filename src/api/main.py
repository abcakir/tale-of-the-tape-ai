from fastapi import FastAPI

app = FastAPI(title="TaleOfTheTape API")

@app.get("/")
def health_check():
    return {"status": "OK", "message": "API is running!"}