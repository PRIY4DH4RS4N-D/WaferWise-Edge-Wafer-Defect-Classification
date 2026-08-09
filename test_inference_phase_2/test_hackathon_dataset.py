# =========================================================
# Project : Wafer Defect Classification
# Module  : Confidence-Based Model Testing (STRICT MODE)
# =========================================================
# Pure inference only
# No training / no normalization / production-safe evaluation

import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =========================================================
# SAFE TEE LOGGER
# =========================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except:
                pass

# =========================================================
# CONFIGURATION
# =========================================================
TEST_DATASET_ROOT = "/home/senba/hackathon_test_dataset"
MODEL_PATH = "/home/senba/wafer_training_tvt/outputs/best_model.pth"

IMG_SIZE = 224
CONF_THRESHOLD = 0.60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# OUTPUT SETUP
# =========================================================
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

log_path = f"logs/wafer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_path, "w")
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# =========================================================
# PROJECT HEADER
# =========================================================
print("\n===================================================")
print(" Project : Wafer Defect Classification")
print(" Module  : Confidence-Based Model Testing (STRICT)")
print(" Model   : MobileNetV3-Small")
print("===================================================\n")

# =========================================================
# MODEL CLASSES
# =========================================================
MODEL_CLASSES = [
    "Bridge_defect", "Clean", "Cmp_defect", "Crack_defect",
    "LER_Defect", "Opens_Defect", "Others", "P_cntamn",
    "Stain_def", "Via_Defect"
]

MODEL_TO_DATASET = {
    "Bridge_defect": "Bridge",
    "Cmp_defect": "CMP",
    "Clean": "Clean",
    "Crack_defect": "Crack",
    "LER_Defect": "LER",
    "Opens_Defect": "Open",
    "Via_Defect": "VIA",
    "Others": "Other",
    "P_cntamn": "Particle",
    "Stain_def": "Other"
}

SEMANTIC_OTHER_SET = {"Other", "Particle"}

# =========================================================
# TRANSFORMS (NO NORMALIZATION)
# =========================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# =========================================================
# DATASET
# =========================================================
dataset = datasets.ImageFolder(TEST_DATASET_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
CLASSES = dataset.classes

# =========================================================
# MODEL LOAD
# =========================================================
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(1024, len(MODEL_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================================================
# METRICS STORAGE
# =========================================================
accepted_true = []
accepted_pred = []

total = 0
accepted = 0
strict_correct = 0

per_class_correct = {c: 0 for c in CLASSES}
per_class_total = {c: 0 for c in CLASSES}

print("================ IMAGE-WISE RESULTS ================\n")

# =========================================================
# INFERENCE (STRICT LOGIC)
# =========================================================
with torch.no_grad():
    for idx, (img, label) in enumerate(loader):

        img = img.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

        true_class = CLASSES[label.item()]
        raw_pred = MODEL_CLASSES[pred_idx.item()]
        pred_class = MODEL_TO_DATASET[raw_pred]

        total += 1

        if conf.item() < CONF_THRESHOLD:
            print(
                f"Image {idx+1:04d} | "
                f"Actual: {true_class:<10} | "
                f"Predicted: {pred_class:<10} | "
                f"Confidence: {conf.item()*100:6.2f}% | "
                f"Decision: REJECT"
            )
            continue

        accepted += 1
        accepted_true.append(true_class)
        accepted_pred.append(pred_class)
        per_class_total[true_class] += 1

        correct = False
        if pred_class == true_class:
            correct = True
        elif true_class == "Other" and pred_class in SEMANTIC_OTHER_SET:
            correct = True

        if correct:
            strict_correct += 1
            per_class_correct[true_class] += 1

        print(
            f"Image {idx+1:04d} | "
            f"Actual: {true_class:<10} | "
            f"Predicted: {pred_class:<10} | "
            f"Confidence: {conf.item()*100:6.2f}% | "
            f"Decision: ACCEPT"
        )

# =========================================================
# FINAL RESULTS
# =========================================================
coverage = (accepted / total) * 100 if total > 0 else 0
accuracy = (strict_correct / accepted) * 100 if accepted > 0 else 0

overall_precision = precision_score(
    accepted_true, accepted_pred,
    average="weighted", zero_division=0
) * 100

overall_recall = recall_score(
    accepted_true, accepted_pred,
    average="weighted", zero_division=0
) * 100

class_precision = precision_score(
    accepted_true, accepted_pred,
    labels=CLASSES, average=None, zero_division=0
)

class_recall = recall_score(
    accepted_true, accepted_pred,
    labels=CLASSES, average=None, zero_division=0
)

print("\n================ FINAL RESULTS =================\n")
print(f"Total Images       : {total}")
print(f"Accepted Images    : {accepted}")
print(f"Coverage           : {coverage:.2f} %")
print(f"Overall Accuracy   : {accuracy:.2f} %")
print(f"Overall Precision  : {overall_precision:.2f} %")
print(f"Overall Recall     : {overall_recall:.2f} %\n")

print("Class-wise Accuracy:")
for cls in CLASSES:
    if per_class_total[cls] > 0:
        acc = (per_class_correct[cls] / per_class_total[cls]) * 100
        print(f"{cls:<10} : {acc:.2f} %")
    else:
        print(f"{cls:<10} : N/A")

print("\nClass-wise Precision:")
for cls, p in zip(CLASSES, class_precision):
    print(f"{cls:<10} : {p*100:.2f} %")

print("\nClass-wise Recall:")
for cls, r in zip(CLASSES, class_recall):
    print(f"{cls:<10} : {r*100:.2f} %")

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(accepted_true, accepted_pred, labels=CLASSES)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES
)
plt.title("Confusion Matrix (Strict Mode)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# =========================================================
# PRECISION–RECALL BAR PLOT
# =========================================================
plt.figure(figsize=(12, 6))
x = range(len(CLASSES))
plt.bar(x, class_precision, width=0.4, label="Precision")
plt.bar([i + 0.4 for i in x], class_recall, width=0.4, label="Recall")
plt.xticks([i + 0.2 for i in x], CLASSES, rotation=45)
plt.ylabel("Score")
plt.title("Class-wise Precision & Recall")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/precision_recall.png")
plt.close()

print("\nSaved Files:")
print("outputs/confusion_matrix.png")
print("outputs/precision_recall.png")
print("\n================ END ================\n")

# =========================================================
# RESTORE STDOUT
# =========================================================
sys.stdout = original_stdout
log_file.close()

print(f"Log file saved at: {log_path}")
print("Execution completed successfully")

