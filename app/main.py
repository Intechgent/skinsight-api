"""FastAPI app exposing the SkinSight classifier."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.model import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("skinsight")

MAX_BYTES = 5 * 1024 * 1024
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["skinsight"] = load_model()
    logger.info("model loaded")
    yield
    models.clear()


app = FastAPI(title="SkinSight API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "skinsight" in models}


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 5 MB")

    start = time.perf_counter()
    try:
        result = predict(models["skinsight"], image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a readable image")
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info("predicted %s (%.3f) in %.1f ms", result["label"], result["confidence"], latency_ms)
    return {**result, "latency_ms": round(latency_ms, 1),
            "disclaimer": "Research demo, not a medical device."}
