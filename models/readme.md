# 🧠 Model Artifacts

This folder contains all trained and deployment-ready model files generated
during the development of the wafer defect classification system.

These artifacts support **training continuity**, **ONNX-based inference**, and
**edge deployment workflows**.

---

## 📁 Files Included

- **best_model.pth**  
  Final trained PyTorch model checkpoint selected using early stopping.  
  Used for final evaluation, ONNX export, and further fine-tuning if required.

- **training_metrics_tvt.pkl**  
  Serialized file containing training and validation metrics.  
  Used for plotting accuracy, loss curves, and performance analysis.

- **wafer_mobilenetv3_tvt.onnx**  
  ONNX-format model exported from the trained PyTorch network.  
  Enables platform-independent inference and edge deployment.

- **wafer_mobilenetv3_tvt.onnx.data**  
  External weight file generated during ONNX export due to model size.  
  Required for correct loading and execution of the ONNX model.

> ⚠️ **Important:**  
> The `.onnx` and `.onnx.data` files must be kept in the **same directory**
> for inference and deployment.

---

## 🎯 Why ONNX?

- Framework-agnostic model format  
- Direct compatibility with **NXP eIQ** toolchain  
- Optimized for low-latency, edge AI inference  
- Supports quantization and deployment optimization flows  

---

## 🚀 Deployment Readiness

- **Model size (ONNX):** ~296 KB (< 1 MB)  
- **Inference latency:** ~10 ms per image (CPU)  
- **Architecture:** MobileNetV3-Small (edge-optimized)  
- **Portability:** Ready for NXP eIQ edge deployment  

This lightweight design ensures strong classification performance
under strict edge compute and memory constraints.
