# 🧠 Model Artifacts

This folder contains all trained and deployment-ready model files generated during development.

## 📁 Files Included

- **wafer_mobilenetv3.pth**  
  Trained PyTorch model checkpoint.  
  Used for further training, evaluation, and model export.

- **wafer_mobilenetv3.onnx**  
  ONNX-format model exported from PyTorch.  
  Used for platform-independent inference and edge deployment workflows.

- **wafer_mobilenetv3.onnx.data**  
  External weight file generated during ONNX export due to model size.  
  Required for correct loading and execution of the ONNX model.

> ⚠️ Note: The `.onnx` and `.onnx.data` files must be kept together for inference and deployment.

## 🎯 Why ONNX?

- Enables framework-agnostic deployment
- Compatible with NXP eIQ edge toolchain
- Optimized for low-latency, edge inference

## 🚀 Deployment Readiness

- Model size: < 1 MB exactly ( 293 kb )
- Inference latency: ~10 ms per image
- Edge-compatible and quantization-friendly architecture
