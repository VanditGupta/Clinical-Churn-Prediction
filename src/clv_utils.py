"""
CLV (Customer Lifetime Value) Utilities for Clinical Study Churn Prediction
Computes CLV from predicted churn probabilities and generates predictions with CLV values
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from config import *

def load_trained_model_and_metadata():
    """
    Load the trained model and metadata
    
    Returns:
        tuple: (model, metadata)
    """
    # Load model
    model = lgb.Booster(model_file=str(CHURN_MODEL_FILE))
    
    # Load metadata
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata

def preprocess_new_data(data, label_encoders):
    """
    Preprocess new data using the same encoders as training
    
    Args:
        data (pd.DataFrame): New data to preprocess
        label_encoders (dict): Label encoders from training
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Make a copy to avoid modifying original data
    processed_data = data.copy()
    
    # Encode categorical variables using saved encoders
    for col, encoder in label_encoders.items():
        if col in processed_data.columns:
            # Handle unseen categories by using 'unknown' category
            processed_data[col] = processed_data[col].astype(str)
            processed_data[col] = processed_data[col].map(lambda x: x if x in encoder.classes_ else 'unknown')
            processed_data[col] = encoder.transform(processed_data[col])
    
    return processed_data

def calculate_clv(churn_probability, monthly_stipend, expected_duration_months=None, 
                 discount_rate=0.05, risk_factor=1.0):
    """
    Calculate Customer Lifetime Value (CLV) based on churn probability
    
    Args:
        churn_probability (float): Predicted probability of churning
        monthly_stipend (float): Monthly stipend amount
        expected_duration_months (float): Expected study duration in months
        discount_rate (float): Discount rate for future cash flows
        risk_factor (float): Risk adjustment factor
        
    Returns:
        float: Calculated CLV
    """
    if expected_duration_months is None:
        expected_duration_months = AVERAGE_STUDY_DURATION_MONTHS
    
    # Calculate retention probability
    retention_probability = 1 - churn_probability
    
    # Calculate expected duration based on churn probability
    # Higher churn probability = shorter expected duration
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

def calculate_expected_retention_months(churn_probability, base_duration=AVERAGE_STUDY_DURATION_MONTHS):
    """
    Calculate expected retention months based on churn probability
    
    Args:
        churn_probability (float): Predicted probability of churning
        base_duration (float): Base study duration in months
        
    Returns:
        float: Expected retention months
    """
    retention_probability = 1 - churn_probability
    expected_months = base_duration * retention_probability
    return max(1, expected_months)  # Minimum 1 month

def generate_predictions_with_clv(data_filepath=None, output_filepath=None):
    """
    Generate predictions with CLV values for the clinical study data
    
    Args:
        data_filepath (str): Path to input data file (optional, uses config default if None)
        output_filepath (str): Path to output predictions file (optional, uses config default if None)
        
    Returns:
        pd.DataFrame: Predictions with CLV values
    """
    print("Generating predictions with CLV values...")
    
    # Use default paths if not provided
    if data_filepath is None:
        data_filepath = RAW_DATA_FILE
    if output_filepath is None:
        output_filepath = PREDICTIONS_FILE
    
    # Load model and metadata
    model, metadata = load_trained_model_and_metadata()
    label_encoders = metadata['label_encoders']
    
    # Load data
    data = pd.read_csv(data_filepath)
    print(f"Loaded {len(data)} records for prediction")
    
    # Prepare features (exclude target and participant_id)
    features = data.drop([TARGET_COLUMN, 'participant_id'], axis=1, errors='ignore')
    
    # Preprocess data
    processed_features = preprocess_new_data(features, label_encoders)
    
    # Make predictions
    churn_probabilities = model.predict(processed_features)
    
    # Calculate CLV for each participant
    clv_values = []
    expected_retention_months = []
    
    for i, (churn_prob, monthly_stipend) in enumerate(zip(churn_probabilities, data['monthly_stipend'])):
        # Calculate expected retention months
        retention_months = calculate_expected_retention_months(churn_prob)
        expected_retention_months.append(retention_months)
        
        # Calculate CLV
        clv = calculate_clv(churn_prob, monthly_stipend, retention_months)
        clv_values.append(clv)
    
    # Create predictions DataFrame
    predictions_df = data.copy()
    predictions_df['churn_probability'] = churn_probabilities
    predictions_df['predicted_churn'] = (churn_probabilities > 0.5).astype(int)
    predictions_df['expected_retention_months'] = expected_retention_months
    predictions_df['clv'] = clv_values
    
    # Add risk categories based on churn probability
    predictions_df['risk_category'] = pd.cut(
        churn_probabilities, 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Add CLV categories
    clv_quartiles = predictions_df['clv'].quantile([0.25, 0.5, 0.75])
    predictions_df['clv_category'] = pd.cut(
        predictions_df['clv'],
        bins=[0, clv_quartiles[0.25], clv_quartiles[0.5], clv_quartiles[0.75], float('inf')],
        labels=['Low CLV', 'Medium CLV', 'High CLV', 'Premium CLV']
    )
    
    # Save predictions
    predictions_df.to_csv(output_filepath, index=False)
    print(f"Predictions saved to {output_filepath}")
    
    # Print summary statistics
    print_prediction_summary(predictions_df)
    
    return predictions_df

def print_prediction_summary(predictions_df):
    """
    Print summary statistics of predictions
    
    Args:
        predictions_df (pd.DataFrame): Predictions DataFrame
    """
    print("\nPrediction Summary:")
    print("=" * 50)
    
    # Overall statistics
    print(f"Total participants: {len(predictions_df)}")
    print(f"Average churn probability: {predictions_df['churn_probability'].mean():.3f}")
    print(f"Predicted churn rate: {predictions_df['predicted_churn'].mean():.3f}")
    print(f"Average CLV: ${predictions_df['clv'].mean():,.2f}")
    print(f"Average expected retention: {predictions_df['expected_retention_months'].mean():.1f} months")
    
    # Risk distribution
    print(f"\nRisk Category Distribution:")
    risk_dist = predictions_df['risk_category'].value_counts()
    for category, count in risk_dist.items():
        percentage = (count / len(predictions_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # CLV distribution
    print(f"\nCLV Category Distribution:")
    clv_dist = predictions_df['clv_category'].value_counts()
    for category, count in clv_dist.items():
        percentage = (count / len(predictions_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Top and bottom CLV participants
    print(f"\nTop 5 CLV Participants:")
    top_clv = predictions_df.nlargest(5, 'clv')[['participant_id', 'clv', 'churn_probability', 'monthly_stipend']]
    for _, row in top_clv.iterrows():
        print(f"  {row['participant_id']}: ${row['clv']:,.2f} (Churn: {row['churn_probability']:.3f}, Stipend: ${row['monthly_stipend']:.0f})")
    
    print(f"\nBottom 5 CLV Participants:")
    bottom_clv = predictions_df.nsmallest(5, 'clv')[['participant_id', 'clv', 'churn_probability', 'monthly_stipend']]
    for _, row in bottom_clv.iterrows():
        print(f"  {row['participant_id']}: ${row['clv']:,.2f} (Churn: {row['churn_probability']:.3f}, Stipend: ${row['monthly_stipend']:.0f})")

def analyze_clv_by_features(predictions_df):
    """
    Analyze CLV distribution by different features
    
    Args:
        predictions_df (pd.DataFrame): Predictions DataFrame
    """
    print("\nCLV Analysis by Features:")
    print("=" * 50)
    
    # CLV by study type
    print("\nAverage CLV by Study Type:")
    clv_by_study = predictions_df.groupby('study_type')['clv'].agg(['mean', 'count']).round(2)
    print(clv_by_study)
    
    # CLV by condition
    print("\nAverage CLV by Condition:")
    clv_by_condition = predictions_df.groupby('condition')['clv'].agg(['mean', 'count']).round(2)
    print(clv_by_condition)
    
    # CLV by location
    print("\nAverage CLV by Location:")
    clv_by_location = predictions_df.groupby('location')['clv'].agg(['mean', 'count']).round(2)
    print(clv_by_location)
    
    # CLV by risk category
    print("\nAverage CLV by Risk Category:")
    clv_by_risk = predictions_df.groupby('risk_category')['clv'].agg(['mean', 'count']).round(2)
    print(clv_by_risk)

def generate_clv_report(predictions_df, output_path=None):
    """
    Generate a comprehensive CLV report
    
    Args:
        predictions_df (pd.DataFrame): Predictions DataFrame
        output_path (str): Path to save the report (optional)
    """
    if output_path is None:
        output_path = PREDICTIONS_DIR / "clv_analysis_report.txt"
    
    with open(output_path, 'w') as f:
        f.write("Clinical Study CLV Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total participants: {len(predictions_df)}\n")
        f.write(f"Average CLV: ${predictions_df['clv'].mean():,.2f}\n")
        f.write(f"Median CLV: ${predictions_df['clv'].median():,.2f}\n")
        f.write(f"CLV Standard Deviation: ${predictions_df['clv'].std():,.2f}\n")
        f.write(f"Total CLV Value: ${predictions_df['clv'].sum():,.2f}\n\n")
        
        # Risk analysis
        f.write("Risk Analysis:\n")
        f.write("-" * 15 + "\n")
        risk_analysis = predictions_df.groupby('risk_category').agg({
            'clv': ['mean', 'count', 'sum'],
            'churn_probability': 'mean'
        }).round(3)
        f.write(str(risk_analysis) + "\n\n")
        
        # Feature analysis
        f.write("CLV by Study Type:\n")
        f.write("-" * 20 + "\n")
        study_analysis = predictions_df.groupby('study_type')['clv'].agg(['mean', 'count', 'sum']).round(2)
        f.write(str(study_analysis) + "\n\n")
        
        f.write("CLV by Condition:\n")
        f.write("-" * 18 + "\n")
        condition_analysis = predictions_df.groupby('condition')['clv'].agg(['mean', 'count', 'sum']).round(2)
        f.write(str(condition_analysis) + "\n\n")
        
        # Top and bottom performers
        f.write("Top 10 CLV Participants:\n")
        f.write("-" * 25 + "\n")
        top_10 = predictions_df.nlargest(10, 'clv')[['participant_id', 'clv', 'churn_probability', 'study_type', 'condition']]
        f.write(str(top_10) + "\n\n")
        
        f.write("Bottom 10 CLV Participants:\n")
        f.write("-" * 28 + "\n")
        bottom_10 = predictions_df.nsmallest(10, 'clv')[['participant_id', 'clv', 'churn_probability', 'study_type', 'condition']]
        f.write(str(bottom_10) + "\n")
    
    print(f"CLV analysis report saved to {output_path}")

def main():
    """Main function to generate predictions with CLV"""
    print("Clinical Study CLV Prediction")
    print("=" * 40)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if model exists
    if not CHURN_MODEL_FILE.exists():
        print(f"Model file not found: {CHURN_MODEL_FILE}")
        print("Please run train.py first to train the model.")
        return
    
    # Check if data exists
    if not RAW_DATA_FILE.exists():
        print(f"Data file not found: {RAW_DATA_FILE}")
        print("Please run data_gen.py first to generate the synthetic data.")
        return
    
    # Generate predictions with CLV
    predictions_df = generate_predictions_with_clv()
    
    # Analyze CLV by features
    analyze_clv_by_features(predictions_df)
    
    # Generate comprehensive report
    generate_clv_report(predictions_df)
    
    print("\nCLV prediction completed successfully!")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"CLV analysis report: {PREDICTIONS_DIR}/clv_analysis_report.txt")

if __name__ == "__main__":
    main() 