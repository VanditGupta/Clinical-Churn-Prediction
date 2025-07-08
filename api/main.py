"""
FastAPI application for Clinical Study Churn & CLV Prediction
Provides endpoints for prediction and SHAP-based explainability with async operations and caching
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional
from functools import lru_cache
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import redis.asyncio as redis
from cachetools import TTLCache

from .schemas import (
    PatientInput,
    PredictionResponse,
    ExplanationResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from .utils import load_model_and_explainer, predict_churn_and_clv, explain_prediction

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Study Churn & CLV Prediction API",
    description="API for predicting patient churn and calculating CLV with SHAP explainability, async operations, and caching",
    version="2.0.0",
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
redis_client = None

# In-memory cache for frequent predictions (fallback if Redis is unavailable)
prediction_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
explanation_cache = TTLCache(maxsize=500, ttl=1800)  # 30 minutes TTL

# Cache configuration
CACHE_TTL = 3600  # 1 hour
EXPLANATION_CACHE_TTL = 1800  # 30 minutes


async def get_redis_client():
    """Get Redis client with connection pooling"""
    global redis_client
    if redis_client is None:
        try:
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_client = redis.Redis(
                host=redis_host,
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Test connection
            await redis_client.ping()
            print(f"✅ Redis connection established to {redis_host}")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}. Using in-memory cache only.")
            redis_client = None
    return redis_client


def generate_cache_key(data: Dict[str, Any], prefix: str = "prediction") -> str:
    """Generate a cache key from input data"""
    # Sort the data to ensure consistent keys
    sorted_data = json.dumps(data, sort_keys=True)
    return f"{prefix}:{hashlib.md5(sorted_data.encode()).hexdigest()}"


async def get_cached_result(
    cache_key: str, redis_client: Optional[redis.Redis] = None
) -> Optional[Dict[str, Any]]:
    """Get result from cache (Redis or in-memory)"""
    try:
        # Try Redis first
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
    except Exception as e:
        print(f"Redis cache error: {e}")

    # Fallback to in-memory cache
    return prediction_cache.get(cache_key)


async def set_cached_result(
    cache_key: str,
    data: Dict[str, Any],
    ttl: int = CACHE_TTL,
    redis_client: Optional[redis.Redis] = None,
):
    """Set result in cache (Redis and in-memory)"""
    try:
        # Try Redis first
        if redis_client:
            await redis_client.setex(cache_key, ttl, json.dumps(data))
    except Exception as e:
        print(f"Redis cache error: {e}")

    # Also set in-memory cache as fallback
    prediction_cache[cache_key] = data


async def background_prediction_logging(
    patient_data: Dict[str, Any], prediction_result: Dict[str, Any]
):
    """Background task to log predictions for analytics"""
    try:
        # Simulate async logging to database or analytics service
        await asyncio.sleep(0.1)  # Simulate async operation
        print(
            f"📊 Logged prediction for patient: {patient_data.get('age', 'N/A')} years old"
        )
    except Exception as e:
        print(f"Error logging prediction: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model and SHAP explainer on startup with async operations"""
    global model, explainer, label_encoders
    print("Loading model and SHAP explainer...")

    # Load model in background to avoid blocking startup
    loop = asyncio.get_event_loop()
    model, explainer, label_encoders = await loop.run_in_executor(
        None, load_model_and_explainer
    )

    # Initialize Redis connection
    await get_redis_client()

    print("Model and explainer loaded successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
        print("Redis connection closed")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Clinical Study Churn & CLV Prediction API",
        "version": "2.0.0",
        "features": [
            "async_operations",
            "caching",
            "batch_predictions",
            "background_tasks",
        ],
        "endpoints": {
            "/predict": "Predict churn probability and CLV (cached)",
            "/explain": "Get SHAP explanation for prediction (cached)",
            "/predict/batch": "Batch predictions for multiple patients",
            "/health": "Health check endpoint",
            "/cache/stats": "Cache statistics",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with cache status"""
    redis_client = await get_redis_client()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "redis_connected": redis_client is not None,
        "cache_stats": {
            "prediction_cache_size": len(prediction_cache),
            "explanation_cache_size": len(explanation_cache),
        },
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    redis_client = await get_redis_client()
    redis_info = {}

    if redis_client:
        try:
            redis_info = await redis_client.info()
        except Exception as e:
            redis_info = {"error": str(e)}

    return {
        "in_memory_cache": {
            "prediction_cache_size": len(prediction_cache),
            "explanation_cache_size": len(explanation_cache),
        },
        "redis_cache": redis_info,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_input: PatientInput, background_tasks: BackgroundTasks):
    """
    Predict churn probability and CLV for a patient with caching

    Args:
        patient_input: Patient features
        background_tasks: FastAPI background tasks

    Returns:
        PredictionResponse: Churn probability and CLV estimate
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Generate cache key
        patient_data = patient_input.dict()
        cache_key = generate_cache_key(patient_data, "prediction")

        # Try to get from cache
        redis_client = await get_redis_client()
        cached_result = await get_cached_result(cache_key, redis_client)

        if cached_result:
            # Add background task for logging
            background_tasks.add_task(
                background_prediction_logging, patient_data, cached_result
            )
            return PredictionResponse(**cached_result)

        # Make prediction (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        churn_probability, clv_estimate = await loop.run_in_executor(
            None, predict_churn_and_clv, patient_data, model, label_encoders
        )

        # Create response
        response_data = {
            "churn_probability": churn_probability,
            "clv_estimate": clv_estimate,
            "risk_category": get_risk_category(churn_probability),
            "message": "Prediction completed successfully",
        }

        # Cache the result
        await set_cached_result(cache_key, response_data, CACHE_TTL, redis_client)

        # Add background task for logging
        background_tasks.add_task(
            background_prediction_logging, patient_data, response_data
        )

        return PredictionResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(patient_input: PatientInput):
    """
    Get SHAP explanation for a patient prediction with caching

    Args:
        patient_input: Patient features

    Returns:
        ExplanationResponse: SHAP values and feature importance
    """
    try:
        if model is None or explainer is None:
            raise HTTPException(status_code=500, detail="Model or explainer not loaded")

        # Generate cache key
        patient_data = patient_input.dict()
        cache_key = generate_cache_key(patient_data, "explanation")

        # Try to get from cache
        redis_client = await get_redis_client()
        cached_result = await get_cached_result(cache_key, redis_client)

        if cached_result:
            return ExplanationResponse(**cached_result)

        # Get SHAP explanation (run in thread pool)
        loop = asyncio.get_event_loop()
        shap_values, feature_importance = await loop.run_in_executor(
            None, explain_prediction, patient_data, model, explainer, label_encoders
        )

        # Create response
        response_data = {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "message": "Explanation generated successfully",
        }

        # Cache the result
        await set_cached_result(
            cache_key, response_data, EXPLANATION_CACHE_TTL, redis_client
        )

        return ExplanationResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(batch_request: BatchPredictionRequest):
    """
    Batch predictions for multiple patients with parallel processing

    Args:
        batch_request: Batch of patient features

    Returns:
        BatchPredictionResponse: Predictions for all patients
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Process predictions in parallel
        tasks = []
        for patient_data in batch_request.patients:
            task = asyncio.create_task(process_single_prediction(patient_data))
            tasks.append(task)

        # Wait for all predictions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        predictions = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({"patient_index": i, "error": str(result)})
            else:
                predictions.append(result)

        return BatchPredictionResponse(
            predictions=predictions,
            errors=errors,
            total_patients=len(batch_request.patients),
            successful_predictions=len(predictions),
            failed_predictions=len(errors),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


async def process_single_prediction(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single prediction with caching"""
    # Generate cache key
    cache_key = generate_cache_key(patient_data, "prediction")

    # Try to get from cache
    redis_client = await get_redis_client()
    cached_result = await get_cached_result(cache_key, redis_client)

    if cached_result:
        return cached_result

    # Make prediction
    loop = asyncio.get_event_loop()
    churn_probability, clv_estimate = await loop.run_in_executor(
        None, predict_churn_and_clv, patient_data, model, label_encoders
    )

    # Create result
    result = {
        "patient_data": patient_data,
        "churn_probability": churn_probability,
        "clv_estimate": clv_estimate,
        "risk_category": get_risk_category(churn_probability),
    }

    # Cache the result
    await set_cached_result(cache_key, result, CACHE_TTL, redis_client)

    return result


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
