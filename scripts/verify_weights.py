"""Check the saved weights reproduce the validation accuracy from notebook 07 (80.3%)."""

from pathlib import Path

import torch
import torch.nn as nn
from medmnist import DermaMNIST
from torch.utils.data import DataLoader
from torchvision import models, transforms

WEIGHTS = Path(__file__).resolve().parent.parent / "model" / "skinsight_resnet_finetuned.pt"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_data = DermaMNIST(split="val", download=True, size=224, transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Build the same architecture, then load the saved weights into it
model = models.resnet18()
model.fc = nn.Linear(512, 7)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()

correct = total = 0
with torch.no_grad():
    for images, labels in val_loader:
        preds = model(images).argmax(dim=1)
        correct += (preds == labels.flatten()).sum().item()
        total += labels.size(0)

print(f"validation accuracy: {100 * correct / total:.1f}%")