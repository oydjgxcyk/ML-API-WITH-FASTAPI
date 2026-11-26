"""Training endpoint — POST /api/train."""

from fastapi import APIRouter, HTTPException

from app.schemas.schemas import ErrorResponse, TrainRequest, TrainResponse
from app.services.ml_service import train_model

router = APIRouter(prefix="/api", tags=["Training"])


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train the ML model",
    description="Train a RandomForestClassifier on the Iris dataset with configurable test size and random state.",
    responses={500: {"model": ErrorResponse}},
)
async def train(request: TrainRequest) -> TrainResponse:
    """Train the machine learning model and persist it to disk."""
    try:
        result = train_model(
            test_size=request.test_size,
            random_state=request.random_state,
        )
        return TrainResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
