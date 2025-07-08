# syntax=docker/dockerfile:1
FROM python:3.11.13 AS base
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

# --- FastAPI target (default) ---
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Streamlit target (optional) ---
FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"] 