from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Disaster Tweet Classification API")

model = None
vectorizer = None

# Input schema
class TweetInput(BaseModel):
    text: str

# Load model on startup (IMPORTANT FIX)
@app.on_event("startup")
def load_model():
    global model, vectorizer
    model = joblib.load(
        r"D:\Machine Learning\Project 5 Disaster Tweet Classification\Models\Disaster_tweet_model.pkl"
    )
    vectorizer = joblib.load(
        r"D:\Machine Learning\Project 5 Disaster Tweet Classification\Models\tfid.pkl"
    )
    print("✅ Model and vectorizer loaded successfully")

@app.get("/")
def home():
    return {"message": "Disaster Tweet Classification API is running 🚀"}

@app.post("/predict")
def predict_disaster(data: TweetInput):
    transformed_text = vectorizer.transform([data.text])
    prob = model.predict_proba(transformed_text)[0][1]

    return {
        "tweet": data.text,
        "disaster": bool(prob >= 0.6),
        "confidence": float(round(prob, 2))
    }

