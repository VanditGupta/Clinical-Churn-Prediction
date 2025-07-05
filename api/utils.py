"""
Utility functions for FastAPI backend
Handles model loading, predictions, CLV calculation, and SHAP explanations
"""

import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from typing import Dict, Tuple, List, Any

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config import *

def load_model_and_explainer():
    """
    Load the trained model, metadata, and initialize SHAP explainer
    
    Returns:
        tuple: (model, explainer, label_encoders)
    """
    try:
        # Load model
        model = lgb.Booster(model_file=str(CHURN_MODEL_FILE))
        print(f"Model loaded from {CHURN_MODEL_FILE}")
        
        # Load metadata
        metadata_path = MODELS_DIR / "model_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        label_encoders = metadata['label_encoders']
        print("Metadata loaded successfully")
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        print("SHAP explainer initialized")
        
        return model, explainer, label_encoders
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_patient_data(patient_data: Dict[str, Any], label_encoders: Dict) -> pd.DataFrame:
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

def predict_churn_and_clv(patient_data: Dict[str, Any], model: lgb.Booster, 
                         label_encoders: Dict) -> Tuple[float, float]:
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
    monthly_stipend = patient_data['monthly_stipend']
    clv_estimate = calculate_clv(churn_probability, monthly_stipend)
    
    return churn_probability, clv_estimate

def calculate_clv(churn_probability: float, monthly_stipend: float, 
                 expected_duration_months: float = None, 
                 discount_rate: float = 0.05, risk_factor: float = 1.0) -> float:
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
        monthly_retention_prob = retention_probability ** month
        discounted_value = monthly_stipend * monthly_retention_prob / ((1 + discount_rate) ** month)
        clv += discounted_value
    
    return clv

def explain_prediction(patient_data: Dict[str, Any], model: lgb.Booster, 
                      explainer: shap.TreeExplainer, label_encoders: Dict) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
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
    for feature, value in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        feature_importance.append({
            "feature": feature,
            "shap_value": value,
            "abs_value": abs(value),
            "direction": "positive" if value > 0 else "negative"
        })
    
    return shap_dict, feature_importance

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
        "monthly_stipend": "Monthly incentive amount",
        "contact_frequency": "Number of staff contacts per month",
        "support_group_member": "Whether patient is in support group",
        "language_barrier": "Whether patient has language barriers",
        "device_usage_compliance": "Percentage of device/wearable compliance",
        "survey_score_avg": "Average satisfaction score from surveys"
    } 