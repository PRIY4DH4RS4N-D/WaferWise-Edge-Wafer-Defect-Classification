# 🏋️ Model Training – MobileNetV3-Small

This directory contains the complete PyTorch training pipeline used to train the wafer defect classification model and export it to ONNX for edge deployment.

---

## 📁 Files in this Folder

training / train_wafer_mobilenet.py

- **train_wafer_mobilenet.py**  
  End-to-end training script that:
  - Loads wafer defect images
  - Trains MobileNetV3-Small
  - Evaluates performance
  - Generates confusion matrix & classification report
  - Exports the trained model to ONNX

---

## 🖥️ System Requirements

- OS: Ubuntu 20.04 / 22.04
- Python: 3.9+
- CPU training supported (GPU optional)
- Internet connection (for pretrained weights)

---

## 📦 Step 1: Clone Repository

```bash
git clone https://github.com/PRIY4DH4RS4N-D/WaferWise-Edge-Wafer-Defect-Classification
```
```bash
cd wafer-defect-classification-edge-ai
```
## 🧪 Step 2: Create Virtual Environment

```bash
sudo apt update

sudo apt install python3-pip python3-venv -y

python3 -m venv wafer_ai

source wafer_ai/bin/activate
```
## 📥 Step 3: Install Dependencies
```bash
pip install torch torchvision torchaudio

pip install numpy matplotlib scikit-learn

pip install onnx onnxruntime onnxscript
```
⚠️ All packages must be installed inside the virtual environment.

## 📂 Step 4: Dataset Placement (IMPORTANT)
The dataset must be placed outside or separately and linked via absolute path.

Expected dataset structure:
```text
Iesa_datasets/
├── Bridge_defect/
├── Clean/
├── Cmp_defect/
├── Crack_defect/
├── LER_Defect/
├── Opens_Defect/
├── Others/
├── P_cntamn/
├── Stain_def/
└── Via_Defect/
```
Each folder contains grayscale wafer images.

## ⚙️ Step 5: Training Configuration (Inside Script)
Key settings in train_wafer_mobilenet.py:

Input size: 224 × 224

Channels: Grayscale → 3-channel

Model: MobileNetV3-Small (pretrained)

Classes: 10

Loss: CrossEntropyLoss

Optimizer: Adam (lr = 1e-4)

Epochs: 30

Batch size: 16

Train / Validation split: 80 / 20

## 🚀 Step 6: Run Training
Activate the virtual environment (if not already):

```bash
source wafer_ai/bin/activate
#Run the training script:
```

```bash
python training/train_wafer_mobilenet.py
```
## 📊 Step 7: Training Outputs
During execution, the script will print:

Epoch-wise training accuracy

Final validation accuracy

Per-class precision, recall, F1-score

Confusion matrix

## 💾 Step 8: Generated Model Artifacts
After successful execution, the following files are generated:
```text
training/
├── wafer_mobilenetv3.pth      # PyTorch trained model
├── wafer_mobilenetv3.onnx     # ONNX export for edge deployment
```
These files are later used for:
ONNX inference testing

NXP eIQ Toolkit deployment

INT8 quantization

## 🎯 Expected Performance (Reference)
Training accuracy: ~99%

Validation accuracy: ~82–85%

Inference latency (ONNX, CPU): ~10 ms

## 🧠 Notes for Reproducibility
Folder names define class labels automatically

Class order is alphabetical and must not be changed

Same preprocessing is used during training and inference

ONNX export uses opset 13 for NXP compatibility

✅ Training Status
This script produces a fully trained, evaluated, and edge-ready model suitable for industrial wafer inspection and deployment on embedded platforms.

---
