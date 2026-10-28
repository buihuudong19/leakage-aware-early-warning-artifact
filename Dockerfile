# Container for the pipeline, with the library versions of the reported run.
#   docker build -t leakaware .
#   docker run --rm -v /path/to/olist-csv:/app/data/raw:ro -v /path/to/empty-folder:/app/outputs leakaware --quick
# Mount an empty folder on /app/outputs: a run overwrites the tables, figures and manifest found there.
# Logistic regression reproduces the reported (Windows) scores; XGBoost subsampling differs between platforms.
# Without arguments the container runs the full pipeline (python run_pipeline.py).
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
ENTRYPOINT ["python", "run_pipeline.py"]
