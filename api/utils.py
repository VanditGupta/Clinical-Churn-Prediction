"""
Utility functions for FastAPI backend
Handles model loading, predictions, CLV calculation, and SHAP explanations with async operations and caching
"""

import sys
import os
import asyncio
import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from typing import Dict, Tuple, List, Any, Optional
from functools import lru_cache
import concurrent.futures

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config import *

# Thread pool for CPU-intensive operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


@lru_cache(maxsize=1)
def load_model_and_explainer():
    """
    Load the trained model, metadata, and initialize SHAP explainer
    Cached to avoid reloading on every request

    Returns:
        tuple: (model, explainer, label_encoders)
    """
    try:
        # Load model
        model = lgb.Booster(model_file=str(CHURN_MODEL_FILE))
        print(f"Model loaded from {CHURN_MODEL_FILE}")

        # Load metadata
        metadata_path = MODELS_DIR / "model_metadata.pkl"
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        label_encoders = metadata["label_encoders"]
        print("Metadata loaded successfully")

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        print("SHAP explainer initialized")

        return model, explainer, label_encoders

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


async def load_model_and_explainer_async():
    """
    Async wrapper for loading model and explainer

    Returns:
        tuple: (model, explainer, label_encoders)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, load_model_and_explainer)


def preprocess_patient_data(
    patient_data: Dict[str, Any], label_encoders: Dict
) -> pd.DataFrame:
    """
    Preprocess patient data using the same encoders as training

    Args:
        patient_data: Dictionary of patient features
        label_encoders: Label encoders from training

    Returns:
        pd.DataFrame: Preprocessed features
    """
    # Create DataFrame
    df = pd.DataFrame([patient_data])

    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            # Handle unseen categories
            value = str(df[col].iloc[0])
            if value in encoder.classes_:
                df[col] = encoder.transform([value])
            else:
                # Use most common class for unseen categories
                df[col] = encoder.transform([encoder.classes_[0]])

    return df


async def preprocess_patient_data_async(
    patient_data: Dict[str, Any], label_encoders: Dict
) -> pd.DataFrame:
    """
    Async wrapper for preprocessing patient data

    Args:
        patient_data: Dictionary of patient features
        label_encoders: Label encoders from training

    Returns:
        pd.DataFrame: Preprocessed features
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, preprocess_patient_data, patient_data, label_encoders
    )


def predict_churn_and_clv(
    patient_data: Dict[str, Any], model: lgb.Booster, label_encoders: Dict
) -> Tuple[float, float]:
    """
    Predict churn probability and calculate CLV for a patient

    Args:
        patient_data: Dictionary of patient features
        model: Trained LightGBM model
        label_encoders: Label encoders from training

    Returns:
        tuple: (churn_probability, clv_estimate)
    """
    # Preprocess data
    processed_data = preprocess_patient_data(patient_data, label_encoders)

    # Make prediction
    churn_probability = model.predict(processed_data)[0]

    # Calculate CLV
    monthly_stipend = patient_data["monthly_stipend"]
    clv_estimate = calculate_clv(churn_probability, monthly_stipend)

    return churn_probability, clv_estimate


async def predict_churn_and_clv_async(
    patient_data: Dict[str, Any], model: lgb.Booster, label_encoders: Dict
) -> Tuple[float, float]:
    """
    Async wrapper for churn prediction and CLV calculation

    Args:
        patient_data: Dictionary of patient features
        model: Trained LightGBM model
        label_encoders: Label encoders from training

    Returns:
        tuple: (churn_probability, clv_estimate)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, predict_churn_and_clv, patient_data, model, label_encoders
    )


def calculate_clv(
    churn_probability: float,
    monthly_stipend: float,
    expected_duration_months: float = None,
    discount_rate: float = 0.05,
    risk_factor: float = 1.0,
) -> float:
    """
    Calculate Customer Lifetime Value (CLV) based on churn probability

    Args:
        churn_probability: Predicted probability of churning
        monthly_stipend: Monthly stipend amount
        expected_duration_months: Expected study duration in months
        discount_rate: Discount rate for future cash flows
        risk_factor: Risk adjustment factor

    Returns:
        float: Calculated CLV
    """
    if expected_duration_months is None:
        expected_duration_months = AVERAGE_STUDY_DURATION_MONTHS

    # Calculate retention probability
    retention_probability = 1 - churn_probability

    # Calculate expected duration based on churn probability
    adjusted_duration = expected_duration_months * retention_probability

    # Apply risk factor
    adjusted_duration *= risk_factor

    # Calculate CLV using discounted cash flow
    clv = 0
    for month in range(1, int(adjusted_duration) + 1):
        # Monthly value decreases over time due to churn risk
        monthly_retention_prob = retention_probability**month
        discounted_value = (
            monthly_stipend * monthly_retention_prob / ((1 + discount_rate) ** month)
        )
        clv += discounted_value

    return clv


async def calculate_clv_async(
    churn_probability: float,
    monthly_stipend: float,
    expected_duration_months: float = None,
    discount_rate: float = 0.05,
    risk_factor: float = 1.0,
) -> float:
    """
    Async wrapper for CLV calculation

    Args:
        churn_probability: Predicted probability of churning
        monthly_stipend: Monthly stipend amount
        expected_duration_months: Expected study duration in months
        discount_rate: Discount rate for future cash flows
        risk_factor: Risk adjustment factor

    Returns:
        float: Calculated CLV
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        calculate_clv,
        churn_probability,
        monthly_stipend,
        expected_duration_months,
        discount_rate,
        risk_factor,
    )


def explain_prediction(
    patient_data: Dict[str, Any],
    model: lgb.Booster,
    explainer: shap.TreeExplainer,
    label_encoders: Dict,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Generate SHAP explanation for a patient prediction

    Args:
        patient_data: Dictionary of patient features
        model: Trained LightGBM model
        explainer: SHAP TreeExplainer
        label_encoders: Label encoders from training

    Returns:
        tuple: (shap_values_dict, feature_importance_list)
    """
    # Preprocess data
    processed_data = preprocess_patient_data(patient_data, label_encoders)

    # Get SHAP values
    shap_values = explainer.shap_values(processed_data)

    # Convert to dictionary with feature names
    feature_names = processed_data.columns.tolist()
    shap_dict = {}

    # For binary classification, we want the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    for i, feature in enumerate(feature_names):
        shap_dict[feature] = float(shap_values[0][i])

    # Create feature importance ranking
    feature_importance = []
    for feature, value in sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        feature_importance.append(
            {
                "feature": feature,
                "shap_value": value,
                "abs_value": abs(value),
                "direction": "positive" if value > 0 else "negative",
            }
        )

    return shap_dict, feature_importance


async def explain_prediction_async(
    patient_data: Dict[str, Any],
    model: lgb.Booster,
    explainer: shap.TreeExplainer,
    label_encoders: Dict,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Async wrapper for SHAP explanation generation

    Args:
        patient_data: Dictionary of patient features
        model: Trained LightGBM model
        explainer: SHAP TreeExplainer
        label_encoders: Label encoders from training

    Returns:
        tuple: (shap_values_dict, feature_importance_list)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, explain_prediction, patient_data, model, explainer, label_encoders
    )


async def batch_predict_async(
    patients_data: List[Dict[str, Any]],
    model: lgb.Booster,
    label_encoders: Dict,
    max_concurrent: int = 10,
) -> List[Tuple[float, float]]:
    """
    Perform batch predictions with controlled concurrency

    Args:
        patients_data: List of patient feature dictionaries
        model: Trained LightGBM model
        label_encoders: Label encoders from training
        max_concurrent: Maximum number of concurrent predictions

    Returns:
        List[Tuple[float, float]]: List of (churn_probability, clv_estimate) tuples
    """
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def predict_with_semaphore(patient_data):
        async with semaphore:
            return await predict_churn_and_clv_async(
                patient_data, model, label_encoders
            )

    # Create tasks for all predictions
    tasks = [predict_with_semaphore(patient_data) for patient_data in patients_data]

    # Execute all predictions concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            # Return default values for failed predictions
            processed_results.append((0.5, 0.0))
        else:
            processed_results.append(result)

    return processed_results


def get_feature_descriptions() -> Dict[str, str]:
    """Get descriptions for all features"""
    return {
        "age": "Patient age in years",
        "gender": "Patient gender",
        "income": "Monthly income in USD",
        "location": "Geographic location",
        "study_type": "Clinical trial phase",
        "condition": "Medical condition being studied",
        "visit_adherence_rate": "Percentage of scheduled visits attended",
        "tenure_months": "Number of months in the study",
        "last_visit_gap_days": "Days since last clinic visit",
        "num_medications": "Number of medications prescribed",
        "has_side_effects": "Whether patient experiences side effects",
        "transport_support": "Whether transport support is provided",
        "monthly_stipend": "Monthly stipend amount",
        "contact_frequency": "Frequency of staff contact",
        "support_group_member": "Whether patient is in support group",
        "language_barrier": "Whether there are language barriers",
        "device_usage_compliance": "Compliance with device usage",
        "survey_score_avg": "Average satisfaction survey score",
    }


async def validate_patient_data(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate patient data asynchronously

    Args:
        patient_data: Dictionary of patient features

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    required_fields = [
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

    for field in required_fields:
        if field not in patient_data:
            errors.append(f"Missing required field: {field}")

    # Validate numeric ranges
    if "age" in patient_data and not (18 <= patient_data["age"] <= 85):
        errors.append("Age must be between 18 and 85")

    if "income" in patient_data and not (20000 <= patient_data["income"] <= 150000):
        errors.append("Income must be between $20,000 and $150,000")

    if "visit_adherence_rate" in patient_data and not (
        0.3 <= patient_data["visit_adherence_rate"] <= 1.0
    ):
        errors.append("Visit adherence rate must be between 0.3 and 1.0")

    # Validate categorical values
    valid_genders = ["Male", "Female", "Other"]
    if "gender" in patient_data and patient_data["gender"] not in valid_genders:
        errors.append(f"Gender must be one of: {valid_genders}")

    valid_locations = ["Urban", "Suburban", "Rural"]
    if "location" in patient_data and patient_data["location"] not in valid_locations:
        errors.append(f"Location must be one of: {valid_locations}")

    return len(errors) == 0, errors


def cleanup_executor():
    """Cleanup thread pool executor"""
    executor.shutdown(wait=True)
