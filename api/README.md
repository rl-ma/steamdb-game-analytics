# Steam Staying Power API

## Overview

The model-serving API provides predictions of whether a Steam hit
is likely to retain its players, using launch-day game information.

This initial version establishes the API contract, input validation,
and monitoring endpoints. Predictions are placeholders until the
registered MLflow model is integrated.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Accept game features and return a prediction |
| GET | `/health` | Check API availability and model readiness |
| GET | `/metrics` | Expose metrics for Prometheus |

## POST /predict

The endpoint accepts 13 input fields.

| Field | Type | Description |
|---|---|---|
| name | string | Game name |
| release_date | string | Release date (YYYY-MM-DD) |
| regular_price | float | Regular game price |
| required_age | float | Minimum required age |
| achievements | float | Number of achievements |
| n_languages | float | Supported languages |
| n_audio_languages | float | Supported audio languages |
| n_screenshots | float | Number of screenshots |
| mac | boolean | macOS support |
| linux | boolean | Linux support |
| has_content_notes | boolean | Whether content notes are present |
| genres | string | Comma-separated genres |
| categories | string | Comma-separated categories |

### Example request

```json
{
  "name": "Example Strategy Game",
  "release_date": "2023-06-15",
  "regular_price": 19.99,
  "required_age": 0,
  "achievements": 30,
  "n_languages": 10,
  "n_audio_languages": 2,
  "n_screenshots": 8,
  "mac": true,
  "linux": false,
  "has_content_notes": false,
  "genres": "Strategy,Indie",
  "categories": "Single-player,Steam Achievements"
}
```

### Example response

```json
{
  "probability": 0.5,
  "prediction": true,
  "model_ready": false
}
```

The probability and classification are currently placeholders.
The `model_ready` field indicates whether the actual model is loaded.
Invalid inputs return an HTTP 422 validation error.

## GET /health

```json
{
  "status": "ok",
  "model_ready": false
}
```

## GET /metrics

Exposes Prometheus metrics for:

- API request count
- Request latency
- API errors

## Run locally

From the repository root:

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

Interactive API documentation:

http://localhost:8000/docs

## Planned implementation

- Load `models:/steam-staying-power@champion` from MLflow at startup.
- Configure the tracking server through `MLFLOW_TRACKING_URI`.
- Use the registered model's `decision_threshold` tag.
- Log prediction inputs and results to PostgreSQL.
- Add clear error handling for model and database failures.
- Connect the API to the Streamlit frontend.
- Integrate Prometheus monitoring.
- Add a Dockerfile and pytest tests.
