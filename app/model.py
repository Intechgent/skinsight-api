"""Load the fine-tuned SkinSight ResNet-18 once and run predictions on uploaded images."""
import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "model" / "skinsight_resnet_finetuned.pt"

CLASS_NAMES = [
    "actinic keratoses",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "nevi",
    "vascular",
]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: Path = WEIGHTS_PATH) -> nn.Module:
    model = models.resnet18()
    model.fc = nn.Linear(512, len(CLASS_NAMES))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: nn.Module, image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    batch = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]
    top = int(probs.argmax())
    return {
        "label": CLASS_NAMES[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
    }
