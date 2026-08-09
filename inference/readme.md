# 🔍 Inference – ONNX Model Testing

This folder contains the inference pipeline used to test the trained wafer defect
classification model exported in **ONNX format**.

The goal is to validate **edge-ready inference**, latency, and model correctness
using ONNX Runtime.

---

## 📄 File Description

- **test_onnx_model.py**  
  Python script for running inference using **ONNX Runtime**.  
  The script performs:
  - Model loading  
  - Image preprocessing  
  - Forward inference  
  - Prediction reporting  

**Outputs include:**
- Predicted defect class  
- Confidence score  
- Inference latency (milliseconds)

---

## ⚙️ Inference Workflow

1. Load ONNX model using ONNX Runtime  
2. Preprocess input image  
   - Convert to grayscale  
   - Resize to **224 × 224**  
   - Normalize pixel values  
3. Run forward pass  
4. Display prediction, confidence, and inference time  

---

## 🚀 How to Run

```bash
python test_onnx_model.py --image sample.png
```

Ensure the ONNX model path inside test_onnx_model.py points to the correct
model file in the model/ directory.

# 📊 Sample Output

<div align="center"> <img src="../proof_images/7.jpeg" width="850"/> </div>

# 🎯 Key Highlights

- Real-time inference: ~10 ms per image

- Edge-ready execution: Lightweight ONNX model

- Validated portability: Compatible with NXP eIQ deployment flows

- No GPU dependency: Runs on CPU for edge simulation

# 📌 Notes

- Input images must be grayscale PNG/JPEG

- Resolution handled internally (224 × 224)

- Designed for deployment validation, not training
