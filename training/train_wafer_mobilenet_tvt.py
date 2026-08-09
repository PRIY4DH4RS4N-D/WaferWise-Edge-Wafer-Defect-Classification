# =========================================================
# Train–Validation–Test Training Script
# Wafer Defect Classification using MobileNetV3-Small
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
import numpy as np
import pickle
import os

# =========================================================
# 1. CONFIGURATION
# =========================================================
DATASET_ROOT = "/home/senba/Iesa_datasets_split"

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8

IMAGE_SIZE = 224
NUM_CLASSES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. DATA TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =========================================================
# 3. DATASET LOADING (TRAIN / VAL / TEST)
# =========================================================
train_dataset = datasets.ImageFolder(
    root=f"{DATASET_ROOT}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATASET_ROOT}/val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{DATASET_ROOT}/test",
    transform=eval_transform
)

class_names = train_dataset.classes
print("\nDetected Classes:", class_names)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================================
# 4. MODEL DEFINITION
# =========================================================
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(1024, NUM_CLASSES)
model = model.to(DEVICE)

# =========================================================
# 5. LOSS FUNCTION & OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =========================================================
# 6. METRIC STORAGE (FOR GRAPHS & ANALYSIS)
# =========================================================
train_accuracy_history = []
val_accuracy_history = []

train_loss_history = []
val_loss_history = []

best_train_accuracy = 0.0
best_val_accuracy = 0.0

early_stop_counter = 0

# =========================================================
# 7. TRAINING & VALIDATION LOOP
# =========================================================
print("\n================ TRAINING STARTED ================\n")

for epoch in range(NUM_EPOCHS):

    # ---------------- TRAINING PHASE ----------------
    model.train()

    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_train_acc = train_correct / train_total

    # ---------------- VALIDATION PHASE ----------------
    model.eval()

    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = running_val_loss / len(val_loader)
    epoch_val_acc = val_correct / val_total

    # ---------------- STORE METRICS ----------------
    train_accuracy_history.append(epoch_train_acc)
    val_accuracy_history.append(epoch_val_acc)

    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)

    best_train_accuracy = max(best_train_accuracy, epoch_train_acc)

    # ---------------- PRINT EPOCH SUMMARY ----------------
    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | "
        f"Train Acc: {epoch_train_acc*100:6.2f}% | "
        f"Val Acc: {epoch_val_acc*100:6.2f}% | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f}"
    )

    # ---------------- EARLY STOPPING ----------------
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        early_stop_counter = 0

        torch.save(
            model.state_dict(),
            f"{OUTPUT_DIR}/best_model.pth"
        )
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

print("\n================ TRAINING COMPLETED ================\n")

# =========================================================
# 8. TESTING PHASE (UNSEEN DATA)
# =========================================================
print("================ TESTING PHASE ================\n")

model.load_state_dict(
    torch.load(f"{OUTPUT_DIR}/best_model.pth")
)

model.eval()

test_correct = 0
test_total = 0

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

test_accuracy = test_correct / test_total

# =========================================================
# 9. FINAL RESULTS PRINT (IMPORTANT)
# =========================================================
print("=============== FINAL MODEL PERFORMANCE ===============")
print(f"Best Training Accuracy   : {best_train_accuracy*100:.2f} %")
print(f"Best Validation Accuracy : {best_val_accuracy*100:.2f} %")
print(f"Test Accuracy            : {test_accuracy*100:.2f} %")
print("=======================================================\n")

print("===== TEST CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, target_names=class_names))

cm_test = confusion_matrix(y_true, y_pred)
print("===== TEST CONFUSION MATRIX =====")
print(cm_test)

# =========================================================
# 10. SAVE METRICS FOR GRAPHING
# =========================================================
with open(f"{OUTPUT_DIR}/training_metrics_tvt.pkl", "wb") as f:
    pickle.dump({
        "train_accuracy": train_accuracy_history,
        "val_accuracy": val_accuracy_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_train_accuracy": best_train_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "confusion_matrix_test": cm_test,
        "class_names": class_names
    }, f)

print("\nAll metrics and model saved successfully.")

