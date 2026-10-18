# SkinSight API

Serving the SkinSight skin lesion classifier (ResNet-18, 77.0% validation accuracy on DermaMNIST)
as a REST API. Model training and experiments live in the
[SkinSight repo](https://github.com/Intechgent/SkinSight).

Work in progress: FastAPI endpoint, Docker, ONNX export and CI coming next.

## Run locally

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m uvicorn app.main:app --reload

Place the trained weights at `model/skinsight_resnet_finetuned.pt` (produced by notebook 07 in the
SkinSight repo), then open http://127.0.0.1:8000/docs.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status and whether the model is loaded |
| `/predict` | POST | Upload an image, returns predicted class, confidence, all class probabilities and latency |
