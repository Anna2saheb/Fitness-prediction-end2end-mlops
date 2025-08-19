from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os
from typing import Optional

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_Logistic Regression.pkl")

app = FastAPI()

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fitness prediction API is running"}

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class PersonData(BaseModel):
    age: float
    height_cm: float
    weight_kg: float
    heart_rate: float
    blood_pressure: float
    sleep_hours: Optional[float] = None
    nutrition_quality: float
    activity_index: float
    smokes: str  # 'yes' or 'no'
    gender: str  # 'M' or 'F'

@app.post("/predict")
def predict(data: PersonData):
    try:
        # Calculate BMI
        bmi = data.weight_kg / ((data.height_cm / 100) ** 2)
        
        # Convert categorical to numerical
        smokes_encoded = 1 if data.smokes.lower() == 'yes' else 0
        gender_M = 1 if data.gender.upper() == 'M' else 0
        
        # Handle missing sleep hours
        sleep_hours = data.sleep_hours if data.sleep_hours is not None else 7.0
        
        # Create feature array in correct order
        features = np.array([[
            data.age,
            data.height_cm,
            data.weight_kg,
            data.heart_rate,
            data.blood_pressure,
            sleep_hours,
            data.nutrition_quality,
            data.activity_index,
            smokes_encoded,
            bmi,
            gender_M
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return {"is_fit": bool(prediction)}
        
    except Exception as e:
        return {"error": str(e)}