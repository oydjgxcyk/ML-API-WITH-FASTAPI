"""FastAPI application entry point."""

from fastapi import FastAPI

from app.routes import evaluate, predict, train

# ── Application instance ────────────────────────────────────
app = FastAPI(
    title="ML API with FastAPI",
    description=(
        "A machine learning REST API built with FastAPI and Scikit-learn. "
        "Train a RandomForestClassifier on the Iris dataset, make predictions, "
        "and evaluate model performance — all through simple REST endpoints."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Register routers ────────────────────────────────────────
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(evaluate.router)


# ── Root health-check ────────────────────────────────────────
@app.get("/", tags=["Health"], summary="Health check")
async def root():
    """Simple health-check endpoint."""
    return {
        "status": "healthy",
        "message": "ML API with FastAPI is running. Visit /docs for interactive API documentation.",
    }
