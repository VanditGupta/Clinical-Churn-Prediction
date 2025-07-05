# Clinical Study Churn & CLV Prediction

A comprehensive machine learning project for predicting participant churn and calculating Customer Lifetime Value (CLV) in clinical studies using LightGBM and SHAP for interpretability.

## 🎯 Project Overview

This project helps clinical research organizations:
- **Predict participant churn** using machine learning
- **Calculate CLV** for each participant based on predicted retention
- **Identify high-risk participants** for targeted interventions
- **Optimize resource allocation** based on participant value
- **Explain model decisions** using SHAP for transparency and compliance
- **Analyze feature interactions** to understand churn drivers

## 📁 Project Structure

```
clinical_churn_clv/
├── mlruns/                                # MLflow experiment tracking directory
│   └── 199252106707606244/               # Experiment runs and artifacts
├── data/raw/clinical_data.csv             # 20,000 synthetic patient records
├── models/churn_model.pkl                 # Trained LightGBM model
├── predictions/predictions_with_clv.csv   # Predictions with CLV values
├── visualizations/                        # Comprehensive model analysis plots
│   ├── feature_importance.png            # LightGBM feature importance
│   ├── roc_curve.png                     # ROC curve with AUC score
│   ├── confusion_matrix.png              # Confusion matrix with metrics
│   ├── prediction_distribution.png       # Prediction probability distribution
│   ├── metrics_summary.png               # Performance metrics summary
│   ├── shap_summary.png                  # SHAP beeswarm plot
│   ├── shap_importance.png               # SHAP feature importance
│   ├── shap_dependence_*.png             # SHAP dependence plots
│   ├── shap_waterfall_*.png              # SHAP waterfall plots
│   └── shap_force_*.png                  # SHAP force plots
├── scripts/
│   └── run_with_tracking.py              # MLflow-enabled training script
├── src/
│   ├── data_gen.py         # Generates mock clinical study data
│   ├── train.py            # Trains LightGBM model with MLflow tracking
│   ├── clv_utils.py        # Computes CLV from predicted churn probabilities
│   └── config.py           # Contains constants, file paths, model params
├── requirements.txt        # Python dependencies (includes MLflow)
├── README.md               # This file
└── .gitignore              # Git ignore patterns
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
cd src
python data_gen.py
```

This creates `data/raw/clinical_data.csv` with 20,000 realistic patient records.

### 3. Train the Model with MLflow Tracking

```bash
# Option A: Run with MLflow tracking (recommended)
python scripts/run_with_tracking.py

# Option B: Run without MLflow tracking
cd src
python train.py
```

The MLflow-enabled training will:
- Log all hyperparameters and metrics
- Save the model to MLflow registry
- Track all artifacts (visualizations, predictions)
- Enable experiment comparison

### 4. View MLflow Results

```bash
# Start MLflow UI
mlflow ui

# Open browser to: http://localhost:5000
# Navigate to experiment: "clinical_churn_prediction"
```

### 5. Generate CLV Predictions

```bash
cd src
python clv_utils.py
```

This creates `predictions/predictions_with_clv.csv` with churn probabilities and CLV values.

## 📊 Dataset Features

The synthetic dataset includes 20 features:

### Demographics
- `participant_id`: Unique identifier
- `age`: Patient age (18-85)
- `gender`: Male/Female/Other
- `income`: Monthly income in USD
- `location`: Urban/Suburban/Rural

### Study Information
- `study_type`: Phase I/II/III
- `condition`: Diabetes, Hypertension, Cardiovascular Disease, etc.
- `tenure_months`: Number of months in study
- `monthly_stipend`: Incentive received by participant

### Engagement Metrics
- `visit_adherence_rate`: % of scheduled visits attended
- `last_visit_gap_days`: Days since last clinic visit
- `contact_frequency`: Outreach per month by staff
- `device_usage_compliance`: % of wearable/device compliance
- `survey_score_avg`: Average satisfaction score

### Clinical Factors
- `num_medications`: Number of medications prescribed
- `has_side_effects`: Binary (1 = yes)
- `support_group_member`: Binary

### Support Factors
- `transport_support`: Binary (1 = transport provided)
- `language_barrier`: Binary

### Target Variables
- `churned`: Target (1 = dropped out, 0 = active)
- `clv`: Derived label = monthly_stipend × expected retention months

## 🔧 Scripts Overview

### `data_gen.py`
- Generates 20,000 synthetic patient records
- Creates realistic correlations between features
- Implements churn probability based on feature interactions
- Saves data to `data/raw/clinical_data.csv`

### `train.py`
- Loads and preprocesses the clinical data
- Trains a LightGBM model for churn prediction
- Performs cross-validation and model evaluation
- Generates comprehensive visualizations and SHAP analysis
- Outputs:
  - `models/churn_model.pkl` (trained model)
  - `models/model_metadata.pkl` (encoders and metadata)
  - `models/model_metrics.txt` (performance metrics)
  - `visualizations/feature_importance.png` (feature importance plot)
  - `visualizations/roc_curve.png` (ROC curve)
  - `visualizations/confusion_matrix.png` (confusion matrix)
  - `visualizations/prediction_distribution.png` (prediction distribution)
  - `visualizations/metrics_summary.png` (metrics summary)
  - `visualizations/shap_summary.png` (SHAP beeswarm plot)
  - `visualizations/shap_importance.png` (SHAP feature importance)
  - `visualizations/shap_dependence_*.png` (SHAP dependence plots)
  - `visualizations/shap_waterfall_*.png` (SHAP waterfall plots)
  - `visualizations/shap_force_*.png` (SHAP force plots)

### `clv_utils.py`
- Loads the trained model and makes predictions
- Calculates CLV using discounted cash flow method
- Categorizes participants by risk and CLV levels
- Generates comprehensive analysis reports
- Outputs:
  - `predictions/predictions_with_clv.csv` (predictions with CLV)
  - `predictions/clv_analysis_report.txt` (detailed analysis)

### `config.py`
- Central configuration file with all constants
- File paths, model parameters, data generation ranges
- Categorical options and feature definitions
- Visualization directory configuration

## 📈 Model Performance

The LightGBM model typically achieves:
- **AUC Score**: ~0.64-0.70
- **Precision**: ~0.60-0.65
- **Recall**: ~0.55-0.60
- **F1 Score**: ~0.57-0.62
- **Accuracy**: ~0.60-0.65

## 💰 CLV Calculation

Customer Lifetime Value is calculated using:
```
CLV = Σ(monthly_stipend × retention_probability^month / (1 + discount_rate)^month)
```

Where:
- `retention_probability = 1 - churn_probability`
- `discount_rate = 0.05` (5% annual discount)
- Expected duration is adjusted based on churn risk

## 🎯 Key Insights

The model typically identifies these as important features:
1. **Visit adherence rate** - Strongest predictor of churn
2. **Last visit gap** - Recent engagement critical
3. **Survey satisfaction score** - Participant satisfaction
4. **Device usage compliance** - Technology engagement
5. **Side effects** - Clinical experience impact

## 🔍 SHAP Analysis

The project includes comprehensive SHAP analysis for model interpretability:

### SHAP Visualizations:
- **Beeswarm Plot**: Shows feature impact distribution across all predictions
- **Feature Importance**: Mean absolute SHAP values for feature ranking
- **Dependence Plots**: How individual features affect predictions
- **Waterfall Plots**: Detailed breakdown of individual predictions
- **Force Plots**: Visual explanation of high-risk cases

### Key Benefits:
- **Model Transparency**: Explainable AI for regulatory compliance
- **Feature Interactions**: Understand how features work together
- **Individual Explanations**: Explain predictions for specific participants

## 📊 MLflow Experiment Tracking

The project includes comprehensive MLflow integration for experiment tracking and model management.

### What's Tracked:

#### Parameters
- **Model Parameters**: All LightGBM hyperparameters (learning_rate, max_depth, etc.)
- **Data Configuration**: Dataset size, feature counts, split ratios
- **CLV Configuration**: Study duration, monthly values, discount rates

#### Metrics
- **Performance Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Cross-validation Results**: Best loss, iterations, standard deviation
- **Model Quality**: Training time, convergence metrics

#### Artifacts
- **Model Files**: LightGBM model saved to MLflow registry
- **Visualizations**: All plots and charts (feature importance, SHAP, etc.)
- **Predictions**: CSV files with churn probabilities and CLV values
- **Metadata**: Model configuration and performance summaries

### MLflow Features:

#### Experiment Management
- **Organized Experiments**: All runs grouped under "clinical_churn_prediction"
- **Run Comparison**: Side-by-side comparison of different model versions
- **Version Control**: Track model evolution over time

#### Model Registry
- **Model Versioning**: Automatic versioning of trained models
- **Model Serving**: Easy deployment of registered models
- **Artifact Storage**: Centralized storage of all model artifacts

#### Visualization
- **Interactive UI**: Web-based interface for exploring experiments
- **Metric Tracking**: Real-time monitoring of training progress
- **Artifact Browsing**: Easy access to all generated files

### Using MLflow:

#### Start Training with Tracking
```bash
python scripts/run_with_tracking.py
```

#### View Results
```bash
# Start MLflow UI
mlflow ui

# Navigate to experiment and explore:
# - Model performance metrics
# - Hyperparameter comparisons
# - Generated visualizations
# - Model artifacts
```

#### Compare Experiments
```bash
# Run multiple experiments with different parameters
# Compare results in MLflow UI
# Select best performing model for deployment
```

#### Model Deployment
```bash
# Load model from MLflow registry
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("runs:/<run_id>/clinical_churn_lightgbm")
```
- **Risk Assessment**: Identify and explain high-risk participants

## 🔍 Risk Categories

Participants are categorized into:
- **Low Risk** (0-30% churn probability)
- **Medium Risk** (30-60% churn probability)  
- **High Risk** (60-100% churn probability)

## 💎 CLV Categories

Based on quartiles:
- **Low CLV**: Bottom 25%
- **Medium CLV**: 25-50%
- **High CLV**: 50-75%
- **Premium CLV**: Top 25%

## 🛠️ Customization

### Modify Data Generation
Edit `config.py` to change:
- Number of records (`NUM_RECORDS`)
- Feature ranges (`AGE_RANGE`, `INCOME_RANGE`, etc.)
- Categorical options (`GENDERS`, `LOCATIONS`, etc.)

### Adjust Model Parameters
Modify `LIGHTGBM_PARAMS` in `config.py`:
```python
LIGHTGBM_PARAMS = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    # ... other parameters
}
```

### Customize CLV Calculation
Adjust CLV parameters in `clv_utils.py`:
- `discount_rate`: Time value of money
- `risk_factor`: Risk adjustment multiplier
- `expected_duration_months`: Base study duration

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- lightgbm >= 3.3.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- shap >= 0.41.0

## 🚨 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the `src` directory when running scripts
2. **File Not Found**: Run scripts in order: `data_gen.py` → `train.py` → `clv_utils.py`
3. **Memory Error**: Reduce `NUM_RECORDS` in `config.py` for smaller datasets

### Performance Tips

- Use GPU acceleration for LightGBM if available
- Adjust `num_boost_round` in training for faster/slower training
- Modify cross-validation folds based on dataset size
- SHAP analysis can be computationally intensive - consider using a subset for large datasets

## 📄 License

This project is for educational and research purposes. The synthetic data generation follows realistic clinical study patterns but is not based on real patient data.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 🚀 Advanced Features

### Model Interpretability
- **SHAP Analysis**: Complete model explainability for regulatory compliance
- **Feature Interactions**: Understanding of how clinical factors interact
- **Individual Predictions**: Detailed explanations for each participant

### Visualization Suite
- **Performance Metrics**: Comprehensive model evaluation plots
- **Feature Analysis**: Multiple perspectives on feature importance
- **Prediction Insights**: Distribution and probability analysis

### Clinical Trial Focus
- **Realistic Data**: Synthetic data mimicking real clinical study patterns
- **Risk Stratification**: Participant categorization for targeted interventions
- **Value Optimization**: CLV-based resource allocation strategies

---

**Note**: This project uses synthetic data for demonstration. In real-world applications, ensure compliance with data privacy regulations and obtain necessary approvals for clinical data analysis. 