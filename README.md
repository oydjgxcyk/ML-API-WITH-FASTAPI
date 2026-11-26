# ML API with FastAPI

A professional machine learning REST API built with **FastAPI** and **Scikit-learn**. Train a Random Forest classifier on the Iris dataset, submit input data for predictions, and evaluate model performance — all through clean REST endpoints.

---

## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [API Usage Examples](#api-usage-examples)
- [Interactive Documentation](#interactive-documentation)
- [License](#license)

---

## Features

- **Model Training** — Train a `RandomForestClassifier` on the Iris dataset via a single API call.
- **Predictions** — Submit four numeric features and receive the predicted class with confidence probabilities.
- **Model Evaluation** — Retrieve accuracy, precision, recall, F1-score, and a per-class classification report.
- **Input Validation** — All requests are validated with Pydantic schemas.
- **Model Persistence** — Trained models are saved and loaded using Joblib.
- **Error Handling** — Meaningful HTTP error responses for missing models, invalid inputs, etc.
- **Interactive Docs** — Auto-generated Swagger UI and ReDoc documentation.

---

## Technologies

| Technology    | Purpose                              |
|---------------|--------------------------------------|
| Python 3.9+   | Programming language                 |
| FastAPI       | Web framework for building APIs      |
| Uvicorn       | ASGI server                          |
| Scikit-learn  | Machine learning model training      |
| Pandas        | Data manipulation and preprocessing  |
| NumPy         | Numerical computing                  |
| Pydantic      | Request/response validation          |
| Joblib        | Model serialization & persistence    |

---

## Project Structure

```
API/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application & router registration
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── train.py           # POST /api/train
│   │   ├── predict.py         # POST /api/predict
│   │   └── evaluate.py        # GET  /api/evaluate
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic request & response models
│   ├── services/
│   │   ├── __init__.py
│   │   └── ml_service.py      # ML training, prediction & evaluation logic
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Dataset loading, model save/load helpers
├── models/                    # Saved model artifacts (auto-generated)
├── requirements.txt
├── README.md
└── run.py                     # Uvicorn launcher
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository** (or download the project):

   ```bash
   git clone <repository-url>
   cd API
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Start the development server:

```bash
python run.py
```

The API will be available at **http://localhost:8000**.

---

## API Endpoints

| Method | Endpoint         | Description                                     |
|--------|------------------|-------------------------------------------------|
| GET    | `/`              | Health check                                    |
| POST   | `/api/train`     | Train the ML model                              |
| POST   | `/api/predict`   | Predict class for given features                |
| GET    | `/api/evaluate`  | Evaluate model (accuracy, precision, recall, F1)|

---

## API Usage Examples

### 1. Train the Model

```bash
curl -X POST http://localhost:8000/api/train \
     -H "Content-Type: application/json" \
     -d '{"test_size": 0.2, "random_state": 42}'
```

**Response:**

```json
{
  "message": "Model trained and saved successfully.",
  "accuracy": 1.0,
  "train_samples": 120,
  "test_samples": 30
}
```

### 2. Make a Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Response:**

```json
{
  "predicted_class": "setosa",
  "prediction_probability": {
    "setosa": 0.98,
    "versicolor": 0.01,
    "virginica": 0.01
  }
}
```

### 3. Evaluate the Model

```bash
curl -X GET http://localhost:8000/api/evaluate
```

**Response:**

```json
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "classification_report": {
    "setosa": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 10.0 },
    "versicolor": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 10.0 },
    "virginica": { "precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 10.0 }
  }
}
```

---

## Interactive Documentation

FastAPI auto-generates interactive API documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## License

This project is released for educational and demonstration purposes.
