"""
Configuration file for Clinical Study Churn & CLV Prediction Project
Contains constants, file paths, and model parameters
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "clinical_data.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CHURN_MODEL_FILE = MODELS_DIR / "churn_model.pkl"

# Prediction paths
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions_with_clv.csv"

# Visualization paths
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "clinical_churn_prediction"
MLFLOW_MODEL_NAME = "clinical_churn_lightgbm"

# Random seed for reproducibility
RANDOM_SEED = 42

# Data generation parameters
NUM_RECORDS = 20000
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature names for the dataset
FEATURE_COLUMNS = [
    "participant_id",
    "age",
    "gender",
    "income",
    "location",
    "study_type",
    "condition",
    "visit_adherence_rate",
    "tenure_months",
    "last_visit_gap_days",
    "num_medications",
    "has_side_effects",
    "transport_support",
    "monthly_stipend",
    "contact_frequency",
    "support_group_member",
    "language_barrier",
    "device_usage_compliance",
    "survey_score_avg",
]

TARGET_COLUMN = "churned"
CLV_COLUMN = "clv"

# Categorical features for encoding
CATEGORICAL_FEATURES = [
    "gender",
    "location",
    "study_type",
    "condition",
    "has_side_effects",
    "transport_support",
    "support_group_member",
    "language_barrier",
]

# Numerical features
NUMERICAL_FEATURES = [
    "age",
    "income",
    "visit_adherence_rate",
    "tenure_months",
    "last_visit_gap_days",
    "num_medications",
    "monthly_stipend",
    "contact_frequency",
    "device_usage_compliance",
    "survey_score_avg",
]

# LightGBM model parameters
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

# CLV calculation parameters
AVERAGE_STUDY_DURATION_MONTHS = 24  # Average expected study duration
BASE_MONTHLY_VALUE = 500  # Base monthly value for CLV calculation

# Data generation ranges
AGE_RANGE = (18, 85)
INCOME_RANGE = (20000, 150000)
VISIT_ADHERENCE_RANGE = (0.3, 1.0)
TENURE_RANGE = (1, 36)
LAST_VISIT_GAP_RANGE = (0, 90)
NUM_MEDICATIONS_RANGE = (0, 8)
MONTHLY_STIPEND_RANGE = (100, 1000)
CONTACT_FREQUENCY_RANGE = (1, 8)
DEVICE_COMPLIANCE_RANGE = (0.2, 1.0)
SURVEY_SCORE_RANGE = (1, 10)

# Categorical options
GENDERS = ["Male", "Female", "Other"]
LOCATIONS = ["Urban", "Suburban", "Rural"]
STUDY_TYPES = ["Phase I", "Phase II", "Phase III"]
CONDITIONS = [
    "Diabetes",
    "Hypertension",
    "Cardiovascular Disease",
    "Obesity",
    "Respiratory Disease",
    "Mental Health",
    "Cancer",
    "Autoimmune Disease",
]


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [RAW_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Create MLflow runs directory
    mlruns_dir = PROJECT_ROOT / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Directories created successfully!")
