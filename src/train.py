"""
Training Script for Clinical Study Churn Prediction
Trains a LightGBM model on clinical study data and saves performance metrics
"""

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.lightgbm
from datetime import datetime
import mlflow.models

from config import *

# Set up MLflow tracking and experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the clinical study data

    Args:
        filepath (str): Path to the CSV data file

    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoders)
    """
    print("Loading and preprocessing data...")

    # Load data
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} records")

    # Separate features and target
    X = data.drop([TARGET_COLUMN, "participant_id"], axis=1)
    y = data[TARGET_COLUMN]

    # Encode categorical variables
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Churn rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, label_encoders


def train_lightgbm_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train a LightGBM model for churn prediction

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features (optional)
        y_val (pd.Series): Validation target (optional)

    Returns:
        lgb.Booster: Trained LightGBM model
    """
    print("Training LightGBM model...")

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets = [train_data, val_data]
        valid_names = ["train", "valid"]
    else:
        valid_sets = [train_data]
        valid_names = ["train"]

    # Train model
    model = lgb.train(
        LIGHTGBM_PARAMS,
        train_data,
        valid_sets=valid_sets,
        valid_names=valid_names,
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics

    Args:
        model (lgb.Booster): Trained LightGBM model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target

    Returns:
        dict: Performance metrics
    """
    print("\nEvaluating model performance...")

    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "auc_score": auc_score,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }

    print(f"\nPerformance Metrics:")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance from the trained model

    Args:
        model (lgb.Booster): Trained LightGBM model
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
    """
    # Get feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    # Plot top features
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

    bars = plt.barh(range(len(top_features)), top_features["importance"], color=colors)
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance (Gain)", fontsize=12)
    plt.title(
        f"Top {top_n} Feature Importance for Churn Prediction",
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features["importance"])):
        plt.text(
            bar.get_width() + max(top_features["importance"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            fontsize=10,
        )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save plot
    plot_path = VISUALIZATIONS_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature importance plot saved to {plot_path}")

    return feature_importance


def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve for the model

    Args:
        model (lgb.Booster): Trained LightGBM model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="#2E86AB", lw=3, label=f"ROC curve (AUC = {auc_score:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="#A23B72", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        "Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold"
    )
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add AUC score text
    plt.text(
        0.6,
        0.2,
        f"AUC = {auc_score:.3f}",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    plot_path = VISUALIZATIONS_DIR / "roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC curve plot saved to {plot_path}")


def plot_confusion_matrix(y_test, y_pred, metrics):
    """
    Plot confusion matrix with detailed annotations

    Args:
        y_test (pd.Series): True labels
        y_pred (np.array): Predicted labels
        metrics (dict): Performance metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        cbar_kws={"label": "Count"},
    )

    plt.title("Confusion Matrix - Churn Prediction", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Add metrics text
    metrics_text = f'Accuracy: {metrics["accuracy"]:.3f}\nPrecision: {metrics["precision"]:.3f}\nRecall: {metrics["recall"]:.3f}\nF1-Score: {metrics["f1_score"]:.3f}'
    plt.text(
        0.02,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    plt.tight_layout()

    # Save plot
    plot_path = VISUALIZATIONS_DIR / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix plot saved to {plot_path}")


def plot_prediction_distribution(y_test, y_pred_proba):
    """
    Plot distribution of prediction probabilities

    Args:
        y_test (pd.Series): True labels
        y_pred_proba (np.array): Prediction probabilities
    """
    plt.figure(figsize=(12, 8))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Distribution by actual class
    for label in [0, 1]:
        mask = y_test == label
        ax1.hist(
            y_pred_proba[mask],
            bins=30,
            alpha=0.7,
            label=f'Actual {"Churn" if label else "No Churn"}',
            color="red" if label else "blue",
        )

    ax1.set_xlabel("Predicted Probability", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(
        "Prediction Probability Distribution by Actual Class",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot
    data_to_plot = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
    labels = ["No Churn", "Churn"]
    colors = ["blue", "red"]

    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel("Predicted Probability", fontsize=12)
    ax2.set_title(
        "Prediction Probability Distribution (Box Plot)", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = VISUALIZATIONS_DIR / "prediction_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Prediction distribution plot saved to {plot_path}")


def plot_metrics_summary(metrics):
    """
    Plot a summary of all performance metrics

    Args:
        metrics (dict): Performance metrics
    """
    # Prepare data
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    metric_values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["auc_score"],
    ]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6B5B95"]

    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.title("Model Performance Metrics Summary", fontsize=16, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis="y")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save plot
    plot_path = VISUALIZATIONS_DIR / "metrics_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Metrics summary plot saved to {plot_path}")


def generate_shap_analysis(model, X_train, X_test, feature_names):
    """
    Generate comprehensive SHAP analysis for model interpretability

    Args:
        model (lgb.Booster): Trained LightGBM model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        feature_names (list): List of feature names
    """
    print("Generating SHAP analysis...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test.values,
        feature_names=feature_names,
    )

    # 1. SHAP Summary Plot
    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_explanation, show=False, max_display=15)
    plt.title(
        "SHAP Summary Plot - Feature Impact on Churn Prediction",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    # Save summary plot
    summary_path = VISUALIZATIONS_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {summary_path}")

    # 2. SHAP Bar Plot (Mean Absolute SHAP Values)
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_explanation, show=False, max_display=15)
    plt.title(
        "SHAP Feature Importance (Mean Absolute Impact)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    # Save bar plot
    bar_path = VISUALIZATIONS_DIR / "shap_importance.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP importance plot saved to {bar_path}")

    # 3. SHAP Dependence Plots for Top Features
    top_features = [
        "last_visit_gap_days",
        "visit_adherence_rate",
        "tenure_months",
        "monthly_stipend",
        "survey_score_avg",
    ]

    for i, feature in enumerate(top_features):
        if feature in X_test.columns:
            plt.figure(figsize=(10, 8))
            # Create dependence plot using the correct API
            shap.plots.scatter(shap_explanation[:, feature], show=False)
            plt.title(
                f"SHAP Dependence Plot: {feature}", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            # Save dependence plot
            dep_path = VISUALIZATIONS_DIR / f"shap_dependence_{feature}.png"
            plt.savefig(dep_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SHAP dependence plot for {feature} saved to {dep_path}")

    # 4. SHAP Waterfall Plot for Sample Predictions
    # Select a few interesting cases
    sample_indices = [0, 100, 500, 1000]  # Different types of predictions

    for i, idx in enumerate(sample_indices):
        if idx < len(X_test):
            plt.figure(figsize=(12, 8))
            # Create waterfall plot using the correct API
            shap.plots.waterfall(shap_explanation[idx], show=False)
            plt.title(
                f"SHAP Waterfall Plot - Sample Prediction {i+1}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            # Save waterfall plot
            waterfall_path = VISUALIZATIONS_DIR / f"shap_waterfall_sample_{i+1}.png"
            plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SHAP waterfall plot for sample {i+1} saved to {waterfall_path}")

    # 5. SHAP Force Plot for High-Risk Cases
    # Find cases with high churn probability
    y_pred_proba = model.predict(X_test)
    high_risk_indices = np.where(y_pred_proba > 0.8)[0][:3]  # Top 3 high-risk cases

    for i, idx in enumerate(high_risk_indices):
        plt.figure(figsize=(12, 6))
        # Create force plot using the correct API
        shap.plots.force(shap_explanation[idx], show=False)
        plt.title(
            f"SHAP Force Plot - High-Risk Case {i+1} (Churn Prob: {y_pred_proba[idx]:.3f})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save force plot
        force_path = VISUALIZATIONS_DIR / f"shap_force_highrisk_{i+1}.png"
        plt.savefig(force_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP force plot for high-risk case {i+1} saved to {force_path}")

    return explainer, shap_values


def log_mlflow_artifacts():
    """
    Log all generated artifacts to MLflow
    """
    print("Logging artifacts to MLflow...")

    # Log visualizations
    if VISUALIZATIONS_DIR.exists():
        for viz_file in VISUALIZATIONS_DIR.glob("*.png"):
            mlflow.log_artifact(str(viz_file), "visualizations")
            print(f"Logged visualization: {viz_file.name}")

    # Log model metadata
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    if metadata_path.exists():
        mlflow.log_artifact(str(metadata_path), "model_metadata")
        print(f"Logged model metadata: {metadata_path.name}")

    # Log metrics file
    metrics_path = MODELS_DIR / "model_metrics.txt"
    if metrics_path.exists():
        mlflow.log_artifact(str(metrics_path), "metrics")
        print(f"Logged metrics file: {metrics_path.name}")

    # Log predictions if they exist
    if PREDICTIONS_FILE.exists():
        mlflow.log_artifact(str(PREDICTIONS_FILE), "predictions")
        print(f"Logged predictions: {PREDICTIONS_FILE.name}")


def log_mlflow_parameters():
    """
    Log all model parameters and configuration to MLflow
    """
    print("Logging parameters to MLflow...")

    # Log LightGBM parameters
    mlflow.log_params(LIGHTGBM_PARAMS)

    # Log data configuration
    mlflow.log_params(
        {
            "num_records": NUM_RECORDS,
            "test_size": TEST_SIZE,
            "validation_size": VALIDATION_SIZE,
            "random_seed": RANDOM_SEED,
            "num_features": len(FEATURE_COLUMNS),
            "categorical_features": len(CATEGORICAL_FEATURES),
            "numerical_features": len(NUMERICAL_FEATURES),
        }
    )

    # Log CLV configuration
    mlflow.log_params(
        {
            "average_study_duration_months": AVERAGE_STUDY_DURATION_MONTHS,
            "base_monthly_value": BASE_MONTHLY_VALUE,
        }
    )

    print("Parameters logged successfully")


def log_mlflow_metrics(metrics, cv_results=None):
    """
    Log all performance metrics to MLflow

    Args:
        metrics (dict): Model performance metrics
        cv_results (dict): Cross-validation results
    """
    print("Logging metrics to MLflow...")

    # Log main metrics
    mlflow.log_metrics(
        {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc_score": metrics["auc_score"],
        }
    )

    # Log cross-validation results if available
    if cv_results:
        mlflow.log_metrics(
            {
                "cv_best_loss": cv_results["best_loss"],
                "cv_best_iteration": cv_results["best_iteration"],
                "cv_loss_std": cv_results["loss_std"],
            }
        )

    print("Metrics logged successfully")


def save_model_and_metadata(
    model, label_encoders, metrics, feature_importance, X_train
):
    """
    Save the trained model and metadata

    Args:
        model (lgb.Booster): Trained LightGBM model
        label_encoders (dict): Dictionary of label encoders
        metrics (dict): Performance metrics
        feature_importance (pd.DataFrame): Feature importance data
        X_train (pd.DataFrame): Training features for signature inference
    """
    # Save model
    model.save_model(str(CHURN_MODEL_FILE))
    print(f"Model saved to {CHURN_MODEL_FILE}")

    # Log model to MLflow
    import mlflow.models
    import numpy as np
    import pandas as pd

    # Create a sample input for the model
    sample_data = pd.DataFrame(
        {
            "age": [55],
            "gender": [0],  # Encoded value
            "income": [60000],
            "location": [0],  # Encoded value
            "study_type": [1],  # Encoded value
            "condition": [0],  # Encoded value
            "visit_adherence_rate": [0.7],
            "tenure_months": [12],
            "last_visit_gap_days": [15],
            "num_medications": [3],
            "has_side_effects": [0],  # Encoded value
            "transport_support": [1],  # Encoded value
            "monthly_stipend": [400],
            "contact_frequency": [3.0],
            "support_group_member": [0],  # Encoded value
            "language_barrier": [0],  # Encoded value
            "device_usage_compliance": [0.6],
            "survey_score_avg": [7.0],
        }
    )

    input_example = sample_data.values
    signature = None
    try:
        from mlflow.models.signature import infer_signature

        # Use a small sample of training data for signature inference
        sample_features = X_train.iloc[:1]  # Use first training sample
        sample_prediction = model.predict(sample_features)
        signature = infer_signature(sample_features, sample_prediction)
    except Exception:
        pass

    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        registered_model_name=MLFLOW_MODEL_NAME,
    )
    print(f"Model logged to MLflow as: {MLFLOW_MODEL_NAME}")

    # Save metadata
    metadata = {
        "label_encoders": label_encoders,
        "metrics": metrics,
        "feature_importance": feature_importance.to_dict("records"),
        "model_params": LIGHTGBM_PARAMS,
        "feature_names": list(feature_importance["feature"]),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
    }

    metadata_path = MODELS_DIR / "model_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Model metadata saved to {metadata_path}")

    # Save metrics to text file
    metrics_path = MODELS_DIR / "model_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Clinical Study Churn Prediction Model - Performance Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"AUC Score: {metrics['auc_score']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")

        f.write("Top 10 Feature Importance:\n")
        f.write("-" * 30 + "\n")
        for i, row in feature_importance.head(10).iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")

    print(f"Model metrics saved to {metrics_path}")


def cross_validate_model(X, y, cv_folds=5):
    """
    Perform cross-validation on the model

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        cv_folds (int): Number of cross-validation folds

    Returns:
        dict: Cross-validation results
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")

    # Create LightGBM dataset
    data = lgb.Dataset(X, label=y)

    # Perform cross-validation
    cv_results = lgb.cv(
        LIGHTGBM_PARAMS,
        data,
        num_boost_round=1000,
        nfold=cv_folds,
        stratified=True,
        shuffle=True,
        seed=RANDOM_SEED,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Extract best scores - use binary_logloss as the metric
    best_loss = min(cv_results["valid binary_logloss-mean"])
    best_iteration = np.argmin(cv_results["valid binary_logloss-mean"]) + 1

    print(f"Cross-validation results:")
    print(f"Best binary_logloss: {best_loss:.4f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Loss std: {cv_results['valid binary_logloss-stdv'][best_iteration-1]:.4f}")

    return {
        "best_loss": best_loss,
        "best_iteration": best_iteration,
        "loss_std": cv_results["valid binary_logloss-stdv"][best_iteration - 1],
    }


def main():
    """Main function to train the churn prediction model"""
    print("Clinical Study Churn Prediction - Model Training")
    print("=" * 50)

    # Ensure directories exist
    ensure_directories()

    # Check if data file exists
    if not RAW_DATA_FILE.exists():
        print(f"Data file not found: {RAW_DATA_FILE}")
        print("Please run data_gen.py first to generate the synthetic data.")
        return

    # Start MLflow run
    with mlflow.start_run(
        run_name=f"clinical_churn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        print(f"MLflow run started: {mlflow.active_run().info.run_id}")

        # Log parameters
        log_mlflow_parameters()

        # Load and preprocess data
        X_train, X_test, y_train, y_test, label_encoders = load_and_preprocess_data(
            RAW_DATA_FILE
        )

        # Perform cross-validation
        cv_results = cross_validate_model(X_train, y_train)

        # Train final model
        model = train_lightgbm_model(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        log_mlflow_metrics(metrics, cv_results)

        # Get predictions for plotting
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Generate all visualizations
        print("\nGenerating visualizations...")

        # Plot feature importance
        feature_importance = plot_feature_importance(model, X_train.columns)

        # Plot ROC curve
        plot_roc_curve(model, X_test, y_test)

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, metrics)

        # Plot prediction distribution
        plot_prediction_distribution(y_test, y_pred_proba)

        # Plot metrics summary
        plot_metrics_summary(metrics)

        # Generate SHAP analysis
        explainer, shap_values = generate_shap_analysis(
            model, X_train, X_test, X_train.columns
        )

        # Save model and metadata
        save_model_and_metadata(
            model, label_encoders, metrics, feature_importance, X_train
        )

        # Log all artifacts
        log_mlflow_artifacts()

        # Log run info
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("dataset_size", len(X_train) + len(X_test))
        mlflow.set_tag("features", len(X_train.columns))

        print(f"\nMLflow run completed: {mlflow.active_run().info.run_id}")

    print("\nTraining completed successfully!")
    print(f"Model saved to: {CHURN_MODEL_FILE}")
    print(f"All visualizations saved to: {VISUALIZATIONS_DIR}/")
    print("Generated plots:")
    print(f"  - Feature importance: {VISUALIZATIONS_DIR}/feature_importance.png")
    print(f"  - ROC curve: {VISUALIZATIONS_DIR}/roc_curve.png")
    print(f"  - Confusion matrix: {VISUALIZATIONS_DIR}/confusion_matrix.png")
    print(
        f"  - Prediction distribution: {VISUALIZATIONS_DIR}/prediction_distribution.png"
    )
    print(f"  - Metrics summary: {VISUALIZATIONS_DIR}/metrics_summary.png")
    print("SHAP Analysis:")
    print(f"  - SHAP summary: {VISUALIZATIONS_DIR}/shap_summary.png")
    print(f"  - SHAP importance: {VISUALIZATIONS_DIR}/shap_importance.png")
    print(f"  - SHAP dependence plots: {VISUALIZATIONS_DIR}/shap_dependence_*.png")
    print(f"  - SHAP waterfall plots: {VISUALIZATIONS_DIR}/shap_waterfall_sample_*.png")
    print(f"  - SHAP force plots: {VISUALIZATIONS_DIR}/shap_force_highrisk_*.png")


if __name__ == "__main__":
    main()
