from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Movie Like Prediction API")

model = joblib.load("model.joblib")

class InputData(BaseModel):
    user_id: int
    movie_id: int
    age: int
    gender: int
    occupation: int

@app.get("/")
def home():
    return {"message": "Movie ML API Running"}

@app.post("/predict")
def predict(data: InputData):

    features = [[
        data.user_id,
        data.movie_id,
        data.age,
        data.gender,
        data.occupation
    ]]

    prob = model.predict_proba(features)[0][1]

    return {
        "will_like": bool(prob > 0.5),
        "confidence": float(prob)
    }