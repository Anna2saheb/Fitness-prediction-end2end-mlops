import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_Logistic Regression.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_fitness(data: dict):
    """Helper function for predictions"""
    # Calculate BMI
    bmi = data['weight_kg'] / ((data['height_cm'] / 100) ** 2)
    
    # Encode categorical features
    smokes_encoded = 1 if data['smokes'].lower() == 'yes' else 0
    gender_M = 1 if data['gender'].upper() == 'M' else 0
    
    # Handle missing sleep hours
    sleep_hours = data.get('sleep_hours', 7.0)  # Default to 7 if not provided
    
    # Create feature array
    features = np.array([[
        data['age'],
        data['height_cm'],
        data['weight_kg'],
        data['heart_rate'],
        data['blood_pressure'],
        sleep_hours,
        data['nutrition_quality'],
        data['activity_index'],
        smokes_encoded,
        bmi,
        gender_M
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    return bool(model.predict(features_scaled)[0])