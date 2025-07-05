# Port Configuration Guide

## Overview

The Clinical Study Churn Prediction project uses three main services, each running on a dedicated port:

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| FastAPI Backend | 8000 | http://localhost:8000 | REST API for predictions and explanations |
| Streamlit Frontend | 8501 | http://localhost:8501 | Interactive web dashboard |
| MLflow UI | 8080 | http://localhost:8080 | Model experiment tracking and visualization |

## Quick Start

### Option 1: Automated Startup (Recommended)
```bash
# Check port availability first
python scripts/check_ports.py

# Start all services
python scripts/start_services.py
```

### Option 2: Manual Startup
```bash
# Terminal 1: FastAPI Backend
uvicorn api.main:app --reload

# Terminal 2: Streamlit Frontend  
streamlit run app/dashboard.py

# Terminal 3: MLflow UI
mlflow ui --port 8080
```

## Port Configuration Details

### FastAPI Backend (Port 8000)
- **File**: `api/main.py`
- **Configuration**: `uvicorn.run(app, host="0.0.0.0", port=8000)`
- **CORS**: Configured for `http://localhost:8501` (Streamlit)
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /predict` - Churn prediction
  - `POST /explain` - SHAP explanations
  - `GET /docs` - Interactive API documentation

### Streamlit Frontend (Port 8501)
- **File**: `app/dashboard.py`
- **Configuration**: `API_BASE_URL = "http://localhost:8000"`
- **Features**: Interactive forms, real-time predictions, SHAP visualizations

### MLflow UI (Port 8080)
- **Configuration**: `mlflow ui --port 8080`
- **Purpose**: Model experiment tracking, run comparison, artifact browsing

## Troubleshooting

### Port Conflicts

#### Check Port Availability
```bash
python scripts/check_ports.py
```

#### Stop Conflicting Processes (macOS/Linux)
```bash
# Stop FastAPI (port 8000)
lsof -ti:8000 | xargs kill -9

# Stop Streamlit (port 8501)
lsof -ti:8501 | xargs kill -9

# Stop MLflow (port 8080)
lsof -ti:8080 | xargs kill -9
```

#### Alternative Ports
If the default ports are unavailable, you can use different ports:

```bash
# FastAPI on different port
uvicorn api.main:app --reload --port 8001

# Streamlit on different port
streamlit run app/dashboard.py --server.port 8502

# MLflow on different port
mlflow ui --port 8081
```

**Note**: If you change FastAPI port, update `API_BASE_URL` in `app/dashboard.py`

### Common Issues

1. **FastAPI not starting**
   - Check if port 8000 is available
   - Ensure all dependencies are installed
   - Check for syntax errors in API code

2. **Streamlit not connecting to FastAPI**
   - Verify FastAPI is running on localhost:8000
   - Check CORS configuration in FastAPI
   - Ensure network connectivity

3. **MLflow UI not accessible**
   - Check if port 8080 is available
   - Verify MLflow is properly installed
   - Check for existing MLflow processes

## Service Dependencies

```
Streamlit Frontend (8501)
    ↓ (API calls)
FastAPI Backend (8000)
    ↓ (model predictions)
MLflow UI (8080) ← Independent service
```

## Development Workflow

1. **Start Development Environment**
   ```bash
   python scripts/start_services.py
   ```

2. **Make Code Changes**
   - FastAPI: Auto-reloads on changes
   - Streamlit: Auto-reloads on changes
   - MLflow: No code changes needed

3. **Monitor Services**
   - Check service health: `python scripts/check_ports.py`
   - View logs in respective terminal windows
   - Use browser to access service URLs

4. **Stop Services**
   - Press `Ctrl+C` in the startup script
   - Or stop individual processes manually

## Production Considerations

### Port Security
- Consider using reverse proxy (nginx) for production
- Implement proper authentication and authorization
- Use HTTPS in production environments

### Service Management
- Use process managers like `systemd` or `supervisord`
- Implement proper logging and monitoring
- Set up health checks and auto-restart

### Scaling
- FastAPI supports multiple workers with `uvicorn --workers`
- Streamlit can be deployed with multiple instances
- MLflow supports distributed tracking servers

## Configuration Files

### Key Configuration Locations
- `api/main.py` - FastAPI app configuration
- `app/dashboard.py` - Streamlit app and API URL
- `scripts/start_services.py` - Service manager configuration
- `scripts/check_ports.py` - Port checking utility

### Environment Variables
Consider using environment variables for port configuration:
```bash
export FASTAPI_PORT=8000
export STREAMLIT_PORT=8501
export MLFLOW_PORT=8080
```

## Support

For additional help:
1. Check the main README.md for detailed instructions
2. Use `python scripts/check_ports.py` for port diagnostics
3. Review service logs for error messages
4. Ensure all dependencies are properly installed 