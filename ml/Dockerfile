FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt jupyterlab \
    && useradd --create-home --uid 1000 ml

COPY preprocess.py .
COPY --chown=ml:ml notebooks/ ./notebooks/

USER ml
WORKDIR /app/notebooks
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
