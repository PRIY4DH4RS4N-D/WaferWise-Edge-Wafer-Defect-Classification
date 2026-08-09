# 📁 Wafer Defect Dataset

This dataset is prepared for the **IESA DeepTech Hackathon 2026** under the problem statement  
**Edge-AI Defect Classification for Semiconductor Images**.

📂 **Dataset Link:**  
🔗 https://drive.google.com/drive/folders/1jaYOw0kGByYc47ywAbBTTPccdnOv3Ki9

The dataset contains grayscale wafer/die inspection images curated to train a  
**lightweight, edge-deployable CNN** for real-time defect classification.

---

## 📂 Dataset Structure

```text
dataset/
├── train/
│   ├── clean/
│   ├── bridge/
│   ├── cmp/
│   ├── open/
│   ├── ler/
│   ├── stain/
│   ├── crack/
│   ├── particle_contam/
│   ├── via/
│   └── others/
├── val/
│   └── (same class folders as train)
├── test/
    └── (same class folders as train)
```

# 🧪 Defect Classes

| Class               | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| **Clean**           | Defect-free wafer regions used as baseline reference                     |
| **Bridge**          | Unintended electrical connections between adjacent metal lines           |
| **CMP**             | Surface scratches or dishing caused during Chemical Mechanical Polishing |
| **Open**            | Broken or incomplete interconnects leading to open circuits              |
| **LER**             | Line Edge Roughness affecting critical dimension control                 |
| **Stain**           | Chemical residue or discoloration from processing steps                  |
| **Crack**           | Structural cracks caused by mechanical or thermal stress                 |
| **Particle Contam** | Foreign particle contamination on wafer surface                          |
| **Via**             | Blocked, missing, or malformed vias impacting vertical interconnects     |
| **Others**          | Rare, ambiguous, or unclassified defect patterns                         |


## 🎯 Why These 10 Classes Were Selected

- Represents commonly observed, high-impact semiconductor fabrication defects

- Covers both systematic and random defect types

- Classes are visually distinguishable, reducing label ambiguity

- Includes Clean class for robust defect vs non-defect learning

- Others class improves model generalization to unseen defects

- Balanced to achieve high accuracy under edge compute constraints

## 🔍 About the **Others** Class

The **Others** category is intentionally designed to improve model robustness and real-world usability.  
It includes:

- **Ambiguous defects** that do not clearly belong to a single class  
- **Partial or edge defects** appearing near wafer boundaries  
- **Imaging artifacts** such as noise, illumination variation, or focus distortion  

This design helps the model:
- Avoid forced misclassification  
- Generalize better to unseen fab conditions  
- Handle real inspection uncertainty at the edge


## 🖼️ Image Characteristics

- **Format:** PNG  
- **Color Space:** Grayscale (single-channel)  
- **Resolution:** **224 × 224 pixels**  
- **Source:** Public semiconductor inspection images collected from research papers and online repositories  
- **Preprocessing:** Resizing, normalization, and controlled augmentations (rotation, mild blur)

## ⚙️ Dataset Design Goals

- Enable real-time inference on edge devices

- Support ONNX export and NXP eIQ compatibility

- Balance model accuracy, size, and latency

- Reflect real fab inspection scenarios

