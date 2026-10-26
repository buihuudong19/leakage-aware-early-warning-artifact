# Container for the pipeline, with the library versions of the reported run.
#   docker build -t leakaware .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" leakaware --quick
# Without arguments the container runs the full pipeline (python run_pipeline.py).
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
ENTRYPOINT ["python", "run_pipeline.py"]
