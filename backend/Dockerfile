# syntax=docker/dockerfile:1
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/spam_classifier_linear_svm.joblib

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 api

COPY app/ ./app/

USER api
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS deploy
COPY --from=model --chmod=0644 /spam_classifier_linear_svm.joblib /models/spam_classifier_linear_svm.joblib

# Keep the default build usable with a locally mounted model.
FROM runtime AS local
