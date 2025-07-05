#!/usr/bin/env python3
"""
MLflow-enabled training script for Clinical Study Churn Prediction
Launches training with comprehensive experiment tracking
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import mlflow
import mlflow.lightgbm
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def setup_mlflow():
    """Setup MLflow tracking and experiment"""
    print("Setting up MLflow experiment tracking...")
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # Get experiment info
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment:
        print(f"Experiment ID: {experiment.experiment_id}")
    else:
        print("Creating new experiment...")

def main():
    """Main function to run training with MLflow tracking"""
    print("=" * 60)
    print("Clinical Study Churn Prediction - MLflow Tracking")
    print("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Import and run training
    from train import main as train_main
    
    print("\nStarting MLflow experiment run...")
    print("All metrics, parameters, and artifacts will be logged to MLflow")
    print("-" * 60)
    
    # Run training (MLflow logging is handled in train.py)
    train_main()
    
    print("\n" + "=" * 60)
    print("Training completed with MLflow tracking!")
    print("=" * 60)
    print("\nTo view results in MLflow UI:")
    print(f"1. Navigate to: {project_root}")
    print("2. Run: mlflow ui --port 8080")
    print("3. Open browser to: http://localhost:8080")
    print("\nTo compare runs:")
    print("1. Run: mlflow ui --port 8080")
    print("2. Select experiment: clinical_churn_prediction")
    print("3. Compare different runs and their metrics")

if __name__ == "__main__":
    main() 