import os
import glob
import joblib
import numpy as np
import pandas as pd

# Stable feature order used during training
FEATURE_ORDER = [
    "age",
    "height_cm",
    "weight_kg",
    "heart_rate",
    "blood_pressure",
    "sleep_hours",
    "nutrition_quality",
    "activity_index",
    "smokes",
    "BMI",
    "gender_M",
]

# Resolve project root and models dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Pick the first "best_model_*.pkl"
CANDIDATES = sorted(glob.glob(os.path.join(MODELS_DIR, "best_model_*.pkl")))
if not CANDIDATES:
    raise FileNotFoundError(f"No best model found in {MODELS_DIR}. Expected a file like best_model_LogisticRegression.pkl")
MODEL_PATH = CANDIDATES[0]

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Missing scaler at {SCALER_PATH}. Ensure your training pipeline saved it.")

# Load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def _ensure_bmi(payload: dict) -> dict:
    data = payload.copy()
    if data.get("BMI") is None:
        h_m = float(data["height_cm"]) / 100.0
        data["BMI"] = float(data["weight_kg"]) / (h_m ** 2)
    return data

def _to_dataframe(payload: dict) -> pd.DataFrame:
    """Create a one-row DataFrame with training-time feature order."""
    ordered = {k: payload[k] for k in FEATURE_ORDER}
    return pd.DataFrame([ordered])

def predict_fitness(person) -> dict:
    """
    person: instance of PersonData (pydantic model)
    returns: {"fit": bool, "probability": float}
    """
    data_dict = person.dict()
    data_dict = _ensure_bmi(data_dict)

    X_df = _to_dataframe(data_dict)
    # scale with the same scaler as training
    X_scaled = scaler.transform(X_df)

    # predict
    pred = model.predict(X_scaled)[0]
    # probability if available
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_scaled)[0][1])
    else:
        proba = None

    return {"fit": bool(pred), "probability": proba}
