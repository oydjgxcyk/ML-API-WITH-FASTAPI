"""Utility helpers for dataset loading and model persistence."""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.joblib")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.joblib")
TARGET_NAMES_PATH = os.path.join(MODEL_DIR, "target_names.joblib")


def ensure_model_dir() -> None:
    """Create the models directory if it does not exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────

def load_dataset() -> Tuple[pd.DataFrame, pd.Series, list]:
    """Load the Iris dataset and return features, labels, and target names.

    Returns:
        Tuple containing:
        - X: DataFrame of shape (150, 4) with feature columns.
        - y: Series of integer labels.
        - target_names: List of class-name strings.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = list(iris.target_names)
    return X, y, target_names


# ── Model persistence ───────────────────────────────────────

def save_model(model: object) -> None:
    """Serialize and save the trained model to disk."""
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)


def load_model() -> object:
    """Load the trained model from disk.

    Raises:
        FileNotFoundError: If no trained model file exists.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Please train the model first via POST /api/train."
        )
    return joblib.load(MODEL_PATH)


def save_test_data(X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Persist the test split so the evaluate endpoint can use it later."""
    ensure_model_dir()
    joblib.dump({"X_test": X_test, "y_test": y_test}, TEST_DATA_PATH)


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the persisted test split.

    Raises:
        FileNotFoundError: If no test data file exists.
    """
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            "No test data found. Please train the model first via POST /api/train."
        )
    data = joblib.load(TEST_DATA_PATH)
    return data["X_test"], data["y_test"]


def save_target_names(target_names: list) -> None:
    """Persist target class names for use in prediction/evaluation."""
    ensure_model_dir()
    joblib.dump(target_names, TARGET_NAMES_PATH)


def load_target_names() -> list:
    """Load persisted target class names.

    Raises:
        FileNotFoundError: If no target names file exists.
    """
    if not os.path.exists(TARGET_NAMES_PATH):
        raise FileNotFoundError(
            "No target names found. Please train the model first via POST /api/train."
        )
    return joblib.load(TARGET_NAMES_PATH)
