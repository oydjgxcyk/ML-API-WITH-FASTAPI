"""Evaluation endpoint — GET /api/evaluate."""

from fastapi import APIRouter, HTTPException

from app.schemas.schemas import ErrorResponse, EvaluateResponse
from app.services.ml_service import evaluate_model

router = APIRouter(prefix="/api", tags=["Evaluation"])


@router.get(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate the trained model",
    description="Evaluate the trained model on the held-out test set. Returns accuracy, precision, recall, F1-score, and a per-class classification report.",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def evaluate() -> EvaluateResponse:
    """Return evaluation metrics for the trained model."""
    try:
        result = evaluate_model()
        return EvaluateResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
