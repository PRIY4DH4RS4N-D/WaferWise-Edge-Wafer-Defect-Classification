# 🔍 Inference – ONNX Model Testing

This folder contains the inference pipeline used to test the trained wafer defect classification model in ONNX format.

## 📄 File Description

- **test_onnx_model.py**  
  Python script for running real-time inference using ONNX Runtime.  
  It loads the exported ONNX model, preprocesses an input wafer image, and outputs:
  - Predicted defect class  
  - Confidence score  
  - Inference latency (ms)

## ⚙️ Inference Workflow

1. Load ONNX model using ONNX Runtime  
2. Preprocess input image (grayscale → resize to 224×224 → normalize)  
3. Run forward pass  
4. Display prediction, confidence, and inference time

## 🚀 How to Run

```bash
python test_onnx_model.py --image sample.png
```
# 📊 Output
<div align="center">
  <img src="proof_images/7.jpeg" width="850"/>
</div>

# 🎯 Key Highlights

- Real-time inference (~10 ms per image)

- Lightweight and edge-ready execution

- Validated ONNX model compatibility for deployment workflows


---
