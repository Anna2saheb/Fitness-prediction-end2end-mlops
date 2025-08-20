import os
import pandas as pd
import pytest
import numpy as np
from app.data_preprocessing import DataPreprocessor

# ---------- Fixtures ----------
@pytest.fixture
def sample_df():
    """Return a small valid dataframe for testing."""
    return pd.DataFrame({
        "age": [25, 30, 45],
        "height_cm": [170, 160, 180],
        "weight_kg": [65, 73, 85],
        "heart_rate": [70, 80, 75],
        "blood_pressure": [120, 130, 125],
        "sleep_hours": [7, np.nan, 8],
        "nutrition_quality": [3, 4, 5],
        "activity_index": [6, 7, 8],
        "smokes": ["yes", "no", "Yes"],
        "gender": ["M", "F", "M"],
        "is_fit": [1, 0, 1],
    })


@pytest.fixture
def preprocessor(tmp_path):
    """Create a DataPreprocessor instance with tmp_path."""
    raw_path = tmp_path / "dummy.csv"
    processed_dir = tmp_path / "processed"
    return DataPreprocessor(str(raw_path), str(processed_dir))


# ---------- Tests ----------
def test_clean_data(sample_df, preprocessor):
    df_clean = preprocessor.clean_data(sample_df)

    # smokes should be converted to int
    assert set(df_clean["smokes"].unique()).issubset({0, 1})
    # sleep_hours should not be NaN anymore
    assert df_clean["sleep_hours"].isnull().sum() == 0


def test_detect_outliers_removes(sample_df, preprocessor):
    # Introduce outlier
    sample_df.loc[0, "weight_kg"] = 9999
    df_no_outliers = preprocessor.detect_outliers(sample_df, z_threshold=3)

    assert len(df_no_outliers) < len(sample_df)


def test_feature_engineering(sample_df, preprocessor):
    df_eng = preprocessor.feature_engineering(sample_df)

    # BMI column should exist
    assert "BMI" in df_eng.columns
    # gender_M should exist
    assert "gender_M" in df_eng.columns
    # gender should be dropped
    assert "gender" not in df_eng.columns


def test_split_data(sample_df, preprocessor):
    df_clean = preprocessor.clean_data(sample_df)
    df_eng = preprocessor.feature_engineering(df_clean)
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_eng)

    # Check splits
    assert len(X_train) + len(X_test) == len(df_eng)
    assert len(y_train) + len(y_test) == len(df_eng)


def test_save_processed_data(tmp_path, sample_df, preprocessor):
    df_eng = preprocessor.feature_engineering(preprocessor.clean_data(sample_df))
    X = df_eng.drop(columns=["is_fit"])
    y = df_eng["is_fit"]

    X_train, X_test, y_train, y_test = preprocessor.split_data(df_eng)

    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)

    # Check if files exist
    train_path = os.path.join(preprocessor.processed_dir, "train.csv")
    test_path = os.path.join(preprocessor.processed_dir, "test.csv")
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)


def test_build_preprocessor(preprocessor):
    preproc = preprocessor.build_preprocessor()
    assert preproc is not None
    assert hasattr(preproc, "fit_transform")
