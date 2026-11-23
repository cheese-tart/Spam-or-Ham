from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .predictor import load_pipeline, predict_label
from .schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.pipeline = load_pipeline()
    yield


app = FastAPI(title="Spam Classifier", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> dict[str, str]:
    label = predict_label(request.app.state.pipeline, payload.text, payload.source)
    return {"label": label}
