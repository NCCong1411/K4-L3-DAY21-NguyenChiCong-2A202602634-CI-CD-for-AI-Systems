import os
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET")
MODEL_KEY = os.getenv("MODEL_KEY", "artifacts/current/model.joblib")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/opt/income-api/models/model.joblib"))


def download_model(s3_client=None) -> Path:
    """Download the current production model from S3."""
    if not ARTIFACT_BUCKET:
        raise RuntimeError("ARTIFACT_BUCKET environment variable is required")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = s3_client or boto3.client("s3")
    client.download_file(ARTIFACT_BUCKET, MODEL_KEY, str(MODEL_PATH))
    print(f"Downloaded s3://{ARTIFACT_BUCKET}/{MODEL_KEY} to {MODEL_PATH}")
    return MODEL_PATH


def load_model_bundle(path: Path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError("Model artifact must contain model and threshold")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "model_bundle", None) is None:
        app.state.model_bundle = load_model_bundle(download_model())
    yield


app = FastAPI(title="Adult Income Prediction API", lifespan=lifespan)
app.state.model_bundle = None


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    if app.state.model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")
    bundle = app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    probability = float(bundle["model"].predict_proba([req.features])[0, 1])
    prediction = int(probability >= float(bundle["threshold"]))
    label = "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap"
    return {
        "prediction": prediction,
        "label": label,
        "probability": probability,
        "threshold": float(bundle["threshold"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
