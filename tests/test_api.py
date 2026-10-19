from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_prediction(client):
    with open(SAMPLES / "4_melanoma.png", "rb") as f:
        r = client.post("/predict", files={"file": ("4_melanoma.png", f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400
