# ==========================================
# Model Training & Selection Pipeline (with Hyperparameter Tuning)
# ==========================================
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Optional models
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------
# Data Loader
# ------------------------------
def load_data(processed_dir: str):
    """Load processed train/test data"""
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ------------------------------
# Candidate Models + Search Spaces
# ------------------------------
def get_models_and_params() -> Dict[str, Any]:
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=500, solver="liblinear"),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 300], "max_depth": [5, 10, None]}
        )
    }
    if XGBClassifier:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
            {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        )
    if LGBMClassifier:
        models["LightGBM"] = (
            LGBMClassifier(random_state=42),
            {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [-1, 10]}
        )
    return models


# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else None,
    }
    return metrics


# ------------------------------
# Save Artifacts
# ------------------------------
def save_artifacts(model, scaler, model_name: str, metrics: Dict[str, float]):
    model_path = os.path.join(MODELS_DIR, f"best_model_{model_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"✅ Best model saved at {model_path}")
    logger.info(f"📊 Metrics saved at {metrics_path}")


# ------------------------------
# Main Training Pipeline
# ------------------------------
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data(PROCESSED_DIR)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate models
    models = get_models_and_params()
    results = {}

    logger.info("Starting model training with hyperparameter tuning...")

    best_overall_model = None
    best_overall_name = None
    best_overall_score = -1
    best_overall_metrics = None

    for name, (model, param_grid) in models.items():
        logger.info(f"🔍 Tuning {name}...")
        grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        metrics = evaluate_model(best_model, X_test_scaled, y_test)
        results[name] = metrics

        logger.info(f"{name} best params: {grid.best_params_}")
        logger.info(f"{name} metrics: {metrics}")

        if metrics["accuracy"] > best_overall_score:
            best_overall_score = metrics["accuracy"]
            best_overall_model = best_model
            best_overall_name = name
            best_overall_metrics = metrics

    logger.info(f"🏆 Best Model: {best_overall_name} with Accuracy {best_overall_score:.4f}")

    # Save best artifacts
    save_artifacts(best_overall_model, scaler, best_overall_name, best_overall_metrics)


if __name__ == "__main__":
    main()
