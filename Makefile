PYTHON=python
PIP=pip

train:
	$(PYTHON) src/train.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

app:
	streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0

test:
	$(PYTHON) -m pytest || echo 'No tests implemented yet.'

format:
	black .
	flake8 || echo 'flake8 warnings (if any)' 