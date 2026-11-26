"""Pydantic schemas for request and response validation."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ───────────────────────── Training ─────────────────────────

class TrainRequest(BaseModel):
    """Request body for the model training endpoint."""

    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of data reserved for testing (0.1 – 0.5).",
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility.",
    )


class TrainResponse(BaseModel):
    """Response body returned after successful training."""

    message: str
    accuracy: float
    train_samples: int
    test_samples: int


# ───────────────────────── Prediction ───────────────────────

class PredictRequest(BaseModel):
    """Request body for the prediction endpoint."""

    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "A list of exactly 4 numeric features: "
            "[sepal_length, sepal_width, petal_length, petal_width]."
        ),
    )


class PredictResponse(BaseModel):
    """Response body returned after a successful prediction."""

    predicted_class: str
    prediction_probability: Dict[str, float]


# ───────────────────────── Evaluation ───────────────────────

class EvaluateResponse(BaseModel):
    """Response body returned by the evaluation endpoint."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    classification_report: Dict[str, Dict[str, float]]


# ───────────────────────── Error ────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
