"""
SHAP Explainer utility for LightGBM models
Initializes and manages SHAP TreeExplainer for model interpretability
"""

import shap
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import pickle
from pathlib import Path

from config import *


class SHAPExplainer:
    """SHAP Explainer wrapper for LightGBM models"""

    def __init__(self, model_path: str = None, metadata_path: str = None):
        """
        Initialize SHAP explainer with LightGBM model

        Args:
            model_path: Path to LightGBM model file
            metadata_path: Path to model metadata file
        """
        if model_path is None:
            model_path = str(CHURN_MODEL_FILE)
        if metadata_path is None:
            metadata_path = str(MODELS_DIR / "model_metadata.pkl")

        # Load model
        self.model = lgb.Booster(model_file=model_path)

        # Load metadata
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Get feature names
        self.feature_names = self.model.feature_name()

        print(f"SHAP Explainer initialized with {len(self.feature_names)} features")

    def explain_prediction(
        self, patient_data: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Generate SHAP explanation for a single prediction

        Args:
            patient_data: Dictionary of patient features

        Returns:
            tuple: (shap_values_dict, feature_importance_list)
        """
        # Preprocess data
        processed_data = self._preprocess_data(patient_data)

        # Get SHAP values
        shap_values = self.explainer.shap_values(processed_data)

        # For binary classification, we want the positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Convert to dictionary
        shap_dict = {}
        for i, feature in enumerate(self.feature_names):
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
                    "description": self._get_feature_description(feature),
                }
            )

        return shap_dict, feature_importance

    def explain_batch(
        self, patient_data_list: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, float], List[Dict[str, Any]]]]:
        """
        Generate SHAP explanations for multiple predictions

        Args:
            patient_data_list: List of patient feature dictionaries

        Returns:
            List of (shap_values_dict, feature_importance_list) tuples
        """
        explanations = []

        for patient_data in patient_data_list:
            explanation = self.explain_prediction(patient_data)
            explanations.append(explanation)

        return explanations

    def get_feature_importance_summary(self, n_top: int = 10) -> List[Dict[str, Any]]:
        """
        Get overall feature importance summary from the model

        Args:
            n_top: Number of top features to return

        Returns:
            List of feature importance dictionaries
        """
        # Get model feature importance
        importance = self.model.feature_importance(importance_type="gain")

        # Create feature importance DataFrame
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Return top features
        top_features = importance_df.head(n_top)

        feature_importance = []
        for _, row in top_features.iterrows():
            feature_importance.append(
                {
                    "feature": row["feature"],
                    "importance": float(row["importance"]),
                    "description": self._get_feature_description(row["feature"]),
                }
            )

        return feature_importance

    def _preprocess_data(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess patient data using saved encoders

        Args:
            patient_data: Dictionary of patient features

        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Create DataFrame
        df = pd.DataFrame([patient_data])

        # Encode categorical variables
        label_encoders = self.metadata["label_encoders"]
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

    def _get_feature_description(self, feature: str) -> str:
        """Get description for a feature"""
        descriptions = {
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
            "survey_score_avg": "Average satisfaction score from surveys",
        }

        return descriptions.get(feature, "Feature description not available")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": "LightGBM",
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "model_path": str(CHURN_MODEL_FILE),
            "explainer_type": "TreeExplainer",
        }


def create_shap_explainer(
    model_path: str = None, metadata_path: str = None
) -> SHAPExplainer:
    """
    Factory function to create SHAP explainer

    Args:
        model_path: Path to LightGBM model file
        metadata_path: Path to model metadata file

    Returns:
        SHAPExplainer: Initialized explainer instance
    """
    return SHAPExplainer(model_path, metadata_path)


if __name__ == "__main__":
    # Test the explainer
    explainer = create_shap_explainer()

    # Test with sample data
    sample_patient = {
        "age": 55,
        "gender": "Female",
        "income": 60000,
        "location": "Urban",
        "study_type": "Phase II",
        "condition": "Diabetes",
        "visit_adherence_rate": 0.8,
        "tenure_months": 12,
        "last_visit_gap_days": 15,
        "num_medications": 3,
        "has_side_effects": False,
        "transport_support": True,
        "monthly_stipend": 400,
        "contact_frequency": 3,
        "support_group_member": False,
        "language_barrier": False,
        "device_usage_compliance": 0.7,
        "survey_score_avg": 7.5,
    }

    shap_values, feature_importance = explainer.explain_prediction(sample_patient)

    print("SHAP Values:")
    for feature, value in shap_values.items():
        print(f"  {feature}: {value:.4f}")

    print("\nTop 5 Features by Impact:")
    for i, feature_info in enumerate(feature_importance[:5]):
        print(
            f"  {i+1}. {feature_info['feature']}: {feature_info['shap_value']:.4f} ({feature_info['direction']})"
        )
