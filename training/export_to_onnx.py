import torch
from torchvision import models
import torch.nn as nn

# ---------------- CONFIG ----------------
NUM_CLASSES = 10
MODEL_PATH = "outputs/best_model.pth"
ONNX_PATH = "outputs/wafer_mobilenetv3_tvt.onnx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(1024, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ---------------- DUMMY INPUT ----------------
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# ---------------- EXPORT ----------------
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=13
)

print(f"ONNX model saved at: {ONNX_PATH}")
