"""Prediction endpoint — POST /api/predict."""

from fastapi import APIRouter, HTTPException

from app.schemas.schemas import ErrorResponse, PredictRequest, PredictResponse
from app.services.ml_service import predict

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Make a prediction",
    description="Submit 4 numeric features (sepal length, sepal width, petal length, petal width) to get a predicted Iris class.",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def make_prediction(request: PredictRequest) -> PredictResponse:
    """Return a prediction for the supplied feature vector."""
    try:
        result = predict(request.features)
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
