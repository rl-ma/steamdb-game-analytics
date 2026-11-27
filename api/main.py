from datetime import date

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

app = FastAPI(title="Steam Staying Power API", version="0.1.0")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint"],
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint"],
)


class GameInput(BaseModel):
    name: str = Field(min_length=1)
    release_date: date
    regular_price: float = Field(ge=0)
    required_age: float = Field(ge=0)
    achievements: float = Field(ge=0)
    n_languages: float = Field(ge=0)
    n_audio_languages: float = Field(ge=0)
    n_screenshots: float = Field(ge=0)
    mac: bool
    linux: bool
    has_content_notes: bool
    genres: str
    categories: str


class PredictionResponse(BaseModel):
    probability: float
    prediction: bool
    model_ready: bool


@app.middleware("http")
async def track_requests(request: Request, call_next):
    endpoint = request.url.path

    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        try:
            response = await call_next(request)
        except Exception:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status="500",
            ).inc()
            ERROR_COUNT.labels(endpoint=endpoint).inc()
            raise

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).inc()

    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=endpoint).inc()

    return response


@app.post("/predict", response_model=PredictionResponse)
def predict(game: GameInput):
    """Return a placeholder prediction until MLflow is integrated."""
    probability = 0.5
    threshold = 0.29

    return PredictionResponse(
        probability=probability,
        prediction=probability >= threshold,
        model_ready=False,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": False,
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
