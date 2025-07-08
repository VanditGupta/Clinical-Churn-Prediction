#!/bin/bash

# Test script for Clinical Churn Prediction API
echo "🧪 Testing Clinical Churn Prediction API..."

# Check if API is running
echo "📊 Checking API status..."
if curl -f http://localhost:8000/health 2>/dev/null; then
    echo "✅ API is running and healthy!"
else
    echo "❌ API is not responding. Starting services..."
    docker compose up -d
    echo "⏳ Waiting for services to start..."
    sleep 30
fi

# Test health endpoint
echo "🔍 Testing health endpoint..."
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed"
    exit 1
}

# Test prediction endpoint
echo "🔮 Testing prediction endpoint..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "Male",
    "income": 60000,
    "location": "Urban",
    "study_type": "Phase II",
    "condition": "Diabetes",
    "visit_adherence_rate": 0.7,
    "tenure_months": 12,
    "last_visit_gap_days": 15,
    "num_medications": 3,
    "has_side_effects": false,
    "transport_support": true,
    "monthly_stipend": 400,
    "contact_frequency": 3.0,
    "support_group_member": false,
    "language_barrier": false,
    "device_usage_compliance": 0.6,
    "survey_score_avg": 7.0
  }' || {
    echo "❌ Prediction test failed"
    exit 1
}

echo "✅ All tests passed!" 