"""
Pydantic schemas for FastAPI request and response models
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class PatientInput(BaseModel):
    """Input schema for patient features"""
    
    age: float = Field(..., ge=18, le=85, description="Patient age")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    income: float = Field(..., ge=20000, le=150000, description="Monthly income in USD")
    location: str = Field(..., description="Location: Urban, Suburban, or Rural")
    study_type: str = Field(..., description="Study type: Phase I, Phase II, or Phase III")
    condition: str = Field(..., description="Medical condition")
    visit_adherence_rate: float = Field(..., ge=0.3, le=1.0, description="Visit adherence rate (0.3-1.0)")
    tenure_months: float = Field(..., ge=1, le=36, description="Months in study")
    last_visit_gap_days: float = Field(..., ge=0, le=90, description="Days since last visit")
    num_medications: int = Field(..., ge=0, le=8, description="Number of medications")
    has_side_effects: bool = Field(..., description="Has side effects")
    transport_support: bool = Field(..., description="Transport support provided")
    monthly_stipend: float = Field(..., ge=100, le=1000, description="Monthly stipend")
    contact_frequency: float = Field(..., ge=1, le=8, description="Contact frequency per month")
    support_group_member: bool = Field(..., description="Support group member")
    language_barrier: bool = Field(..., description="Language barrier")
    device_usage_compliance: float = Field(..., ge=0.2, le=1.0, description="Device compliance rate")
    survey_score_avg: float = Field(..., ge=1, le=10, description="Average survey score")

class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    
    churn_probability: float = Field(..., description="Predicted churn probability")
    clv_estimate: float = Field(..., description="Estimated Customer Lifetime Value")
    risk_category: str = Field(..., description="Risk category: Low, Medium, or High")
    message: str = Field(..., description="Response message")

class ExplanationResponse(BaseModel):
    """Response schema for explanation endpoint"""
    
    shap_values: Dict[str, float] = Field(..., description="SHAP values for each feature")
    feature_importance: List[Dict[str, Any]] = Field(..., description="Feature importance ranking")
    message: str = Field(..., description="Response message")

class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    explainer_loaded: bool = Field(..., description="Whether SHAP explainer is loaded")

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    
    patients: List[Dict[str, Any]] = Field(..., description="List of patient features")
    max_concurrent: Optional[int] = Field(10, description="Maximum concurrent predictions")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    predictions: List[Dict[str, Any]] = Field(..., description="List of successful predictions")
    errors: List[Dict[str, Any]] = Field(..., description="List of prediction errors")
    total_patients: int = Field(..., description="Total number of patients in batch")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")

class CacheStatsResponse(BaseModel):
    """Response schema for cache statistics"""
    
    in_memory_cache: Dict[str, int] = Field(..., description="In-memory cache statistics")
    redis_cache: Dict[str, Any] = Field(..., description="Redis cache information") 