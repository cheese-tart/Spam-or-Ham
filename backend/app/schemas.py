from typing import Literal

from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    text: str
    source: Literal["email", "sms"]

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message text must contain at least one non-whitespace character.")
        return value


class PredictionResponse(BaseModel):
    label: Literal["ham", "spam"]
