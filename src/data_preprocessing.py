"""
Data Preprocessing Pipeline for Fitness Prediction Project
"""

import os
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    End-to-end data preprocessing pipeline for fitness prediction data.
    """
    
    def __init__(
        self,
        raw_data_path: str,
        processed_dir: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data with validation checks."""
        logger.info(f"Loading data from {self.raw_data_path}")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Data file not found at {self.raw_data_path}")
            
        df = pd.read_csv(self.raw_data_path)
        
        # Basic validation
        required_columns = {
            'age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure',
            'sleep_hours', 'nutrition_quality', 'activity_index', 'smokes', 
            'gender', 'is_fit'
        }
        
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform data cleaning operations."""
        logger.info("Starting data cleaning")
        
        df_clean = df.copy()
        
        # Standardize smokes column
        df_clean['smokes'] = (
            df_clean['smokes']
            .replace({'no': 0, 'yes': 1, 'No': 0, 'Yes': 1})
            .astype(int)
        )
        
        # Impute missing sleep hours with median
        if df_clean['sleep_hours'].isnull().any():
            median_sleep = df_clean['sleep_hours'].median()
            df_clean['sleep_hours'] = df_clean['sleep_hours'].fillna(median_sleep)
            logger.info(f"Imputed {df_clean['sleep_hours'].isnull().sum()} missing sleep_hours values")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        logger.info("Detecting outliers")
        
        numeric_cols = ['weight_kg', 'height_cm', 'age', 'heart_rate']
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        outlier_mask = (z_scores < z_threshold).all(axis=1)
        
        n_outliers = len(df) - sum(outlier_mask)
        if n_outliers > 0:
            logger.info(f"Removing {n_outliers} outliers (z-score > {z_threshold})")
            df = df[outlier_mask].copy()
            
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        logger.info("Performing feature engineering")
        
        # Calculate BMI
        df['BMI'] = df['weight_kg'] / ((df['height_cm']/100)**2)
        
        # One-hot encode gender
        df['gender_M'] = (df['gender'] == 'M').astype(int)
        df.drop('gender', axis=1, inplace=True)
        
        return df
    
    def build_preprocessor(self) -> ColumnTransformer:
        """Build sklearn preprocessing pipeline."""
        numeric_features = [
            'age', 'height_cm', 'weight_kg', 'heart_rate', 
            'blood_pressure', 'sleep_hours', 'nutrition_quality',
            'activity_index', 'BMI'
        ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_features = ['smokes', 'gender_M']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', 'passthrough', categorical_features)
            ]
        )
        
        return preprocessor
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'is_fit'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        logger.info("Splitting data into train/test sets")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """Save processed data to disk with proper error handling."""
        try:
            train_path = os.path.join(self.processed_dir, "train.csv")
            test_path = os.path.join(self.processed_dir, "test.csv")
            
            # Combine features and target
            train_data = X_train.copy()
            train_data['target'] = y_train
            test_data = X_test.copy()
            test_data['target'] = y_test
            
            # Save to temporary files first
            temp_train_path = train_path + '.tmp'
            temp_test_path = test_path + '.tmp'
            
            train_data.to_csv(temp_train_path, index=False)
            test_data.to_csv(temp_test_path, index=False)
            
            # Atomic rename
            os.replace(temp_train_path, train_path)
            os.replace(temp_test_path, test_path)
            
            logger.info(f"Data successfully saved to:\n- {train_path}\n- {test_path}")
            
        except PermissionError:
            logger.error("Permission denied. Possible solutions:")
            logger.error("1. Close any open CSV files in Excel/other programs")
            logger.error("2. Run the script as Administrator")
            logger.error(f"3. Check permissions for: {self.processed_dir}")
            raise
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def run_pipeline(self) -> None:
        """Execute full preprocessing pipeline."""
        try:
            # 1. Load data
            df = self.load_data()
            
            # 2. Clean data
            df = self.clean_data(df)
            
            # 3. Handle outliers
            df = self.detect_outliers(df)
            
            # 4. Feature engineering
            df = self.feature_engineering(df)
            
            # 5. Split data
            X_train, X_test, y_train, y_test = self.split_data(df)
            
            # 6. Save processed data
            self.save_processed_data(X_train, X_test, y_train, y_test)
            
            logger.info("Data preprocessing completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Configure paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fitness_dataset.csv")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # Run pipeline
    try:
        preprocessor = DataPreprocessor(
            raw_data_path=RAW_DATA_PATH,
            processed_dir=PROCESSED_DIR
        )
        preprocessor.run_pipeline()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)