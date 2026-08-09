import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

# -----------------------------
# CONFIG
# -----------------------------
ONNX_MODEL = "wafer_mobilenetv3.onnx"
TEST_IMAGE = "/home/senba/Downloads/6.png"
IMG_SIZE = 224

CLASS_NAMES = [
    'Bridge_defect', 'Clean', 'Cmp_defect', 'Crack_defect',
    'LER_Defect', 'Opens_Defect', 'Others', 'P_cntamn',
    'Stain_def', 'Via_Defect'
]

# -----------------------------
# TRANSFORMS (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = Image.open(TEST_IMAGE).convert("L")
img = transform(img)
img = img.unsqueeze(0).numpy()

# -----------------------------
# LOAD ONNX MODEL
# -----------------------------
session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# -----------------------------
# INFERENCE + TIMING
# -----------------------------
start_time = time.time()

outputs = session.run(None, {input_name: img})

end_time = time.time()

# -----------------------------
# POST-PROCESS (SOFTMAX)
# -----------------------------
logits = np.squeeze(outputs[0])

# Softmax for confidence
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)

pred_class = np.argmax(probs)
confidence = probs[pred_class] * 100  # percentage

inference_time_ms = (end_time - start_time) * 1000

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("Predicted class :", CLASS_NAMES[pred_class])
print("Confidence      :", round(float(confidence), 2), "%")
print("Inference time  :", round(inference_time_ms, 2), "ms")
