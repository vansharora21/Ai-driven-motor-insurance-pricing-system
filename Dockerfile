FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Trained artifacts are committed to the repo; the API only needs to load them.
RUN mkdir -p models results/logs

EXPOSE 8000

# Serve the FastAPI inference backend. Training/retraining is a separate,
# explicit step (scripts/train.py / scripts/retrain.py) — never run at boot.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]