FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 ml \
    && mkdir -p /app/data /app/artifacts \
    && chown ml:ml /app/data /app/artifacts

COPY preprocess.py .

FROM base AS training
RUN python -m pip install --no-cache-dir boto3
COPY train.py .
USER ml
ENTRYPOINT ["python", "/app/train.py"]

FROM base AS notebook
RUN python -m pip install --no-cache-dir jupyterlab
COPY --chown=ml:ml notebooks/ ./notebooks/

USER ml
WORKDIR /app/notebooks
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
