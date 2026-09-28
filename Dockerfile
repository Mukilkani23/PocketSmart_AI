# PocketSmart AI: one container: FastAPI + models + static frontend.
# Built by Cloud Build via `gcloud run deploy --source .`
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POCKETSMART_CACHE_DIR=/tmp/pocketsmart-cache

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Models are TRAINED AT BUILD TIME from source (seeded, pinned), never committed to git:
#   1. regenerate the synthetic data   2. retrain every model (artifacts only, no --report)
#   3. assert the rebuilt model reproduces the COMMITTED results/metrics.json, else FAIL the build
RUN python -m data.generate \
 && python -m src.evaluate \
 && python -m src.verify_build

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
# Cloud Run injects $PORT
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
