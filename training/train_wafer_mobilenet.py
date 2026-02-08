import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "/home/senba/Iesa_datasets"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# TRANSFORMS (Grayscale → 3-channel)
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)

class_names = dataset.classes
print("Classes:", class_names)

# 80% Train, 20% Validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# MODEL: MobileNetV3-Small
# -----------------------------
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(1024, NUM_CLASSES)
model = model.to(DEVICE)

# -----------------------------
# LOSS & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f} Train Acc: {train_acc:.2f}%")

# -----------------------------
# VALIDATION & CONFUSION MATRIX
# -----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "wafer_mobilenetv3.pth")

# -----------------------------
# EXPORT TO ONNX
# -----------------------------
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    "wafer_mobilenetv3.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("ONNX model exported successfully.")
