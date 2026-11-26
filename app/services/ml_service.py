"""Machine learning service — training, prediction, and evaluation logic."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.utils.helpers import (
    load_dataset,
    load_model,
    load_target_names,
    load_test_data,
    save_model,
    save_target_names,
    save_test_data,
)


def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a RandomForestClassifier on the Iris dataset.

    Args:
        test_size: Fraction of data used for the test split.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with training summary (message, accuracy, sample counts).
    """
    # 1. Load and split the dataset
    X, y, target_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # 2. Train the classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 3. Quick accuracy check on the test split
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))

    # 4. Persist model, test data, and class names
    save_model(model)
    save_test_data(X_test.values, y_test.values)
    save_target_names(target_names)

    return {
        "message": "Model trained and saved successfully.",
        "accuracy": round(accuracy, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


def predict(features: list) -> Dict[str, Any]:
    """Run a prediction for a single sample.

    Args:
        features: List of 4 numeric feature values.

    Returns:
        Dictionary with predicted class label and per-class probabilities.
    """
    model = load_model()
    target_names = load_target_names()

    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]

    return {
        "predicted_class": target_names[int(prediction)],
        "prediction_probability": {
            name: round(float(prob), 4)
            for name, prob in zip(target_names, probabilities)
        },
    }


def evaluate_model() -> Dict[str, Any]:
    """Evaluate the trained model on the held-out test set.

    Returns:
        Dictionary with accuracy, precision, recall, F1-score, and a
        per-class classification report.
    """
    model = load_model()
    target_names = load_target_names()
    X_test, y_test = load_test_data()

    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="macro"))
    recall = float(recall_score(y_test, y_pred, average="macro"))
    f1 = float(f1_score(y_test, y_pred, average="macro"))

    # Detailed per-class report (dict form)
    report: Dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True,
    )

    # Keep only the per-class entries (drop 'accuracy', 'macro avg', etc.)
    per_class = {
        name: {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1-score": round(metrics["f1-score"], 4),
            "support": metrics["support"],
        }
        for name, metrics in report.items()
        if name in target_names
    }

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "classification_report": per_class,
    }
