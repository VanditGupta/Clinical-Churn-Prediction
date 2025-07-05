"""
FastAPI application for Clinical Study Churn & CLV Prediction
Provides endpoints for prediction and SHAP-based explainability
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any

from .schemas import PatientInput, PredictionResponse, ExplanationResponse
from .utils import load_model_and_explainer, predict_churn_and_clv, explain_prediction

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Study Churn & CLV Prediction API",
    description="API for predicting patient churn and calculating CLV with SHAP explainability",
    version="1.0.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and explainer
model = None
explainer = None
label_encoders = None

@app.on_event("startup")
async def startup_event():
    """Load model and SHAP explainer on startup"""
    global model, explainer, label_encoders
    print("Loading model and SHAP explainer...")
    model, explainer, label_encoders = load_model_and_explainer()
    print("Model and explainer loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Clinical Study Churn & CLV Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Predict churn probability and CLV",
            "/explain": "Get SHAP explanation for prediction",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_input: PatientInput):
    """
    Predict churn probability and CLV for a patient
    
    Args:
        patient_input: Patient features
        
    Returns:
        PredictionResponse: Churn probability and CLV estimate
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Make prediction
        churn_probability, clv_estimate = predict_churn_and_clv(
            patient_input.dict(), model, label_encoders
        )
        
        return PredictionResponse(
            churn_probability=churn_probability,
            clv_estimate=clv_estimate,
            risk_category=get_risk_category(churn_probability),
            message="Prediction completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain(patient_input: PatientInput):
    """
    Get SHAP explanation for a patient prediction
    
    Args:
        patient_input: Patient features
        
    Returns:
        ExplanationResponse: SHAP values and feature importance
    """
    try:
        if model is None or explainer is None:
            raise HTTPException(status_code=500, detail="Model or explainer not loaded")
        
        # Get SHAP explanation
        shap_values, feature_importance = explain_prediction(
            patient_input.dict(), model, explainer, label_encoders
        )
        
        return ExplanationResponse(
            shap_values=shap_values,
            feature_importance=feature_importance,
            message="Explanation generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

def get_risk_category(churn_probability: float) -> str:
    """Categorize risk based on churn probability"""
    if churn_probability < 0.3:
        return "Low Risk"
    elif churn_probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 