import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "ml" / "artifacts" / "spam_classifier_linear_svm.joblib"
)


def load_pipeline() -> Pipeline:
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Set MODEL_PATH to the saved pipeline."
        )
    return joblib.load(model_path)


def predict_label(pipeline: Pipeline, text: str, source: str) -> str:
    message = pd.DataFrame({"text": [text], "source": [source]})
    return str(pipeline.predict(message)[0])
