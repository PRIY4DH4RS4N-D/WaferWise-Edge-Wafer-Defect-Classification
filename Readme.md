<!--
██████╗ ██████╗ ██╗██████╗  █████╗ ██╗   ██╗
██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██║   ██║
██████╔╝██████╔╝██║██║  ██║███████║██║   ██║
██╔═══╝ ██╔══██╗██║██║  ██║██╔══██║██║   ██║
██║     ██║  ██║██║██████╔╝██║  ██║╚██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝
-->
<div align="center">
  
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=105&section=header&text=IESA%20Deeptech%20Hackathon%202026%20-%20Waferwise&fontSize=38&fontAlign=50&fontColor=000000" width="100%">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Mono&size=28&pause=1200&color=E65100&center=true&vCenter=true&width=900&lines=Data+%E2%9E%9E+MobileNetV3-Small+%E2%9E%9E+ONNX+%E2%9E%9E+Edge+AI;82%E2%80%9385%25%2B+Test+Accuracy+%7C+~10ms+Inference+%7C+Edge-Ready" />

<p>
  <img src="https://img.shields.io/badge/MobileNetV3-Small%20CNN-ff9800?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-Training-ef5350?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/ONNX-Edge%20Deployment-43a047?style=for-the-badge&logo=onnx"/>
  <img src="https://img.shields.io/badge/Edge--Ready-%3C10ms-ffb300?style=for-the-badge"/>
</p>
</div>

---

## 📌 Project Overview

Our project presents an Edge-AI based wafer defect classification system designed to perform real-time semiconductor inspection directly on edge devices.

The solution leverages a lightweight MobileNetV3-Small CNN model trained on a custom dataset of wafer defect images across 10 classes (including Clean and Other).

The trained model is exported to ONNX format for compatibility with NXP eIQ deployment flow, ensuring portability, low latency (~10 ms inference), and compact model size (<1 MB).

The system addresses latency, bandwidth, and scalability limitations of centralized inspection by enabling efficient on-device AI inference suitable for Industry 4.0 manufacturing environments.

---

## Problem Understanding

Semiconductor fabrication generates massive volumes of inspection images daily. Traditional centralized or manual review systems face:

High analysis latency

Heavy cloud/infrastructure dependency

Bandwidth bottlenecks

Poor scalability for real-time throughput

There is a need for a lightweight, portable, and high-accuracy AI model capable of performing defect classification directly at the edge while maintaining compute efficiency

---

## Dataset Strategy

To meet hackathon requirements and ensure strong generalization, we built a custom wafer inspection dataset.

### 📌 Data Collection

- Collected wafer defect images from publicly available semiconductor datasets and research references

- Included Clean, Other, and 8 distinct defect classes

- Total dataset size: ~1,200+ images (well-balanced)

### 📌 Preprocessing

- Converted all images to grayscale (single-channel)

- Resized to 224 × 224

- Stored in PNG format

Applied controlled augmentations (rotation, mild blur)

### 📌 Dataset Structure

- Dataset organized using structured class-wise directories for training and validation

- Folder-based labeling for supervised learning

- Ensured balanced class distribution

---

## Model Development

### 📌 Architecture Selection

- Selected MobileNetV3-Small for lightweight edge-friendly design  
- Suitable for low-latency and small memory footprint  
- Transfer learning used for faster convergence  

### 📌 Training Setup

- Implemented using PyTorch  
- Multi-class classification (10 classes)  
- Trained with optimizer and cross-entropy loss  
- Batch size: 16 | Epochs: 30  

### 📌 Performance Monitoring

- Tracked training & validation accuracy  
- Monitored loss for stability  
- Generated confusion matrix for class-level evaluation

```mermaid
flowchart LR
    A[Wafer Dataset<br/>~1200+ Images] --> B[Preprocessing<br/>Grayscale • 224x224 • Augmentation]
    B --> C[MobileNetV3-Small<br/>Transfer Learning]
    C --> D[Training<br/>Batch Size 16 • 30 Epochs]
    D --> E[Validation<br/>82–85% Accuracy]
    E --> F[ONNX Export<br/>Edge-Ready Model]
```

---

## 🔹 Evaluation & Performance

Our MobileNetV3-Small based model was optimized for both accuracy and edge efficiency.  
The evaluation results demonstrate strong generalization while maintaining extremely low model size and latency — making it ideal for real-time fab deployment.

### 📊 Model Performance Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | 83–85% |
| Precision (Macro Avg) | ~0.84 |
| Recall (Macro Avg) | ~0.83 |
| F1-Score (Macro Avg) | ~0.83 |
| Model Size (ONNX) | **293 KB** |
| Inference Time | ~10 ms / image |
| Framework | PyTorch → ONNX |

### 📈 Observations

- Confusion matrix shows strong separation across major defect categories.  
- Lightweight model enables real-time inference under strict edge constraints.  
- Extremely compact size (293 KB) ensures easy portability to NXP eIQ flows.

---

## 💡 Innovation

- Lightweight MobileNetV3-Small architecture optimized for edge constraints  
- Extremely compact model size (~293 KB) without sacrificing multi-class performance  
- Designed for direct portability to NXP eIQ deployment flow  
- Edge-first design eliminating dependency on centralized cloud infrastructure  

---

## 🌍 Impact

This solution enables faster and more reliable semiconductor defect inspection by shifting intelligence directly to the edge.

- Reduces manual inspection dependency and human error  
- Enables near real-time wafer analysis (~10 ms/image)  
- Eliminates heavy cloud bandwidth requirements  
- Scales easily across multiple inspection stations  
- Low model size (~293 KB) minimizes hardware cost and memory usage  

By combining accuracy with ultra-lightweight deployment, the system supports Industry 4.0 manufacturing environments with practical, scalable AI.

---

## Artifacts & Links

- 📂 **Dataset (Train/Validation structured)**  
  [[Dataset Download Link](https://drive.google.com/drive/folders/1Atj94_75VKlZoFJyq0if6iPj1cTrWbvc?usp=sharing)]

- 🧠 **Trained ONNX Model (~293 KB)**  
  [[ONNX Model Link](https://drive.google.com/drive/folders/1Atj94_75VKlZoFJyq0if6iPj1cTrWbvc?usp=sharing)]

- 📊 **Evaluation Results (Confusion Matrix & Metrics)**  
  Included in repository under `/results`

