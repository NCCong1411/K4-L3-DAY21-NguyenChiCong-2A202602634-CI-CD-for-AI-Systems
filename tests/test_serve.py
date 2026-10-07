import numpy as np
from fastapi.testclient import TestClient

from src.serve import app


class DummyModel:
    def predict_proba(self, rows):
        return np.array([[0.2, 0.8] for _ in rows])


def test_health_and_score_endpoints():
    app.state.model_bundle = {"model": DummyModel(), "threshold": 0.6}
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post("/score", json={"features": [1] * 10})
        assert response.status_code == 200
        assert response.json()["prediction"] == 1
        assert response.json()["label"] == "thu_nhap_cao"
        assert response.json()["threshold"] == 0.6


def test_score_rejects_wrong_feature_count():
    app.state.model_bundle = {"model": DummyModel(), "threshold": 0.6}
    with TestClient(app) as client:
        response = client.post("/score", json={"features": [1, 2]})
        assert response.status_code == 400
        assert response.json()["detail"] == "Expected 10 features (adult income)"
