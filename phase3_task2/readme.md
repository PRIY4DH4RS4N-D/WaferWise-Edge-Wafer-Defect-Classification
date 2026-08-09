<p align="center">
  <img src="processor-animation.svg" width="900"/>
</p>
<!--
██████╗ ██████╗ ██╗██████╗  █████╗ ██╗   ██╗
██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██║   ██║
██████╔╝██████╔╝██║██║  ██║███████║██║   ██║
██╔═══╝ ██╔══██╗██║██║  ██║██╔══██║██║   ██║
██║     ██║  ██║██║██████╔╝██║  ██║╚██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝
-->
<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=105&section=header&text=🧠%20IESA%20Deeptech%20Hackathon%202026%20-%20WaferWise&fontSize=38&fontAlign=50&fontColor=000000" width="100%">
</div>

---

 - [`prediction/`](./logs/predictions_log.txt) – Prediction done in hotel at 10:25 PM

---

## 📌 Project Overview

- Trained a multi-class wafer defect classification model using 128×128 grayscale input (1-channel).
- Converted the trained model to TensorFlow Lite (.tflite) format.
- Applied INT8 quantization to reduce model size for edge deployment.
- Converted the optimized model into C header files for microcontroller integration.
- Demonstrated a complete Edge AI pipeline from training to embedded inference.
 <div align="center">
  <img src="proof_images/Screenshot from 2026-02-25 13-05-21.png" width="850"/>
</div>

---

## 📊 Model Evaluation – Phase 3 Final (Float32)

- Evaluated on a 10% held-out test set containing 400 grayscale images across 11 wafer defect classes.
- Input shape: 128 × 128 × 1 (Grayscale), Output: 11-class softmax, Model size: 1970.5 KB.
- Achieved 91.25% overall accuracy (365/400) with a 91.3% weighted F1-score.
- Highest-performing classes: CLEAN_CRACK (100%), BRIDGE (100%), VIA (95.1%), CRACK (93.9%).
- Lowest-performing class: CLEAN_LAYER (78.8%), with minor confusion among visually similar defects.
- Confusion matrix shows strong diagonal dominance, indicating stable and reliable classification performance.
 <div align="center">
  <img src="proof_images/WhatsApp Image 2026-02-25 at 3.52.37 PM.jpeg" width="850"/>
</div>
 <div align="center">
  <img src="proof_images/WhatsApp Image 2026-02-25 at 3.52.36 PM.jpeg" width="850"/>
</div>

---

## 📌 Complete ML to Embedded Pipeline

The wafer defect model is trained using grayscale images and converted to TensorFlow Lite format.
It is optimized using INT8 quantization to reduce size and improve embedded performance.
Finally, the model is converted to a C header file and deployed for microcontroller-based inference.
```mermaid
flowchart TB
    subgraph INGEST ["📥  INGESTION"]
        A[(🗄️ Dataset)]
    end

    subgraph TRAIN ["🏋️  TRAINING"]
        B{{⚙️ train.py}} --> C([🧠 Float32 Model])
    end

    subgraph COMPRESS ["⚡  COMPRESSION"]
        D[/🔬 INT8 Quantization/] --> E[(📦 wafer_int8.tflite)]
    end

    subgraph EXPORT ["🔧  EXPORT"]
        F[[🔄 Convert to C Header]] --> G>📄 model_data.h]
    end

    subgraph DEPLOY ["🚀  DEPLOYMENT"]
        H((🚀 Embedded Inference))
    end

    INGEST --> TRAIN --> COMPRESS --> EXPORT --> DEPLOY

    style INGEST   fill:#0d2137,stroke:#00e5ff,stroke-width:2px,color:#00e5ff
    style TRAIN    fill:#1a1000,stroke:#ffc800,stroke-width:2px,color:#ffc800
    style COMPRESS fill:#0f1a0a,stroke:#a8ff3e,stroke-width:2px,color:#a8ff3e
    style EXPORT   fill:#1a0d00,stroke:#ff6b35,stroke-width:2px,color:#ff6b35
    style DEPLOY   fill:#120d1f,stroke:#8264ff,stroke-width:2px,color:#8264ff

    style A fill:#0d2137,stroke:#00e5ff,color:#cceeff
    style B fill:#1a1000,stroke:#ffc800,color:#fff3cc
    style C fill:#1a1000,stroke:#ffc800,color:#fff3cc
    style D fill:#0f1a0a,stroke:#a8ff3e,color:#dfffb0
    style E fill:#0f1a0a,stroke:#a8ff3e,color:#dfffb0
    style F fill:#1a0d00,stroke:#ff6b35,color:#ffd4bb
    style G fill:#1a0d00,stroke:#ff6b35,color:#ffd4bb
    style H fill:#120d1f,stroke:#8264ff,color:#d4c8ff
```
---
## ⚙️ Embedded Deployment Architecture

- The quantized INT8 TFLite model is converted into a C header file (model_data.h).
- The model is integrated into the microcontroller firmware using TFLite Micro runtime.
- A grayscale input image (128×128×1) is converted into a C array (test_image.h).
- The microcontroller loads the model and input data into memory.
- On-device inference is executed without external computation.
- The predicted wafer defect class is generated in real time.

```mermaid
flowchart TB
    subgraph INPUTS ["⚙️  FIRMWARE INPUTS"]
        direction TB
        subgraph FLASH ["🗃️  Flash Memory"]
            M>📄 model_data.h\nINT8 Weights Array]
            I>🖼️ test_image.h\nPixel Buffer Array]
        end
        subgraph HARDWARE ["🔩  Hardware Layer"]
            MCU([🔲 Microcontroller\nClock · GPIO · SRAM])
        end
    end

    subgraph RUNTIME ["⚡  TFLITE MICRO RUNTIME"]
        direction TB
        OP1[/🔁 OpResolver\nRegister Kernels/]
        OP2[/🧩 Interpreter\nBuild Tensor Graph/]
        OP3[/📐 Arena Allocator\nStatic Memory Pool/]
        OP1 --> OP2 --> OP3
    end

    subgraph INFER ["🔍  INFERENCE ENGINE"]
        direction LR
        INV[Invoke\nForward Pass] --> SOFT[Softmax\nActivation]
    end

    subgraph OUTPUT ["🏁  OUTPUT"]
        direction TB
        R1{{✅ Defect\nDetected}}
        R2{{🟢 Wafer\nPass}}
        CONF[(📊 Confidence\nScore)]
    end

    MCU          -->|"drives"| OP1
    M            -->|"load weights"| OP2
    I            -->|"load input tensor"| OP2
    OP3          -->|"ready"| INV
    SOFT         -->|"argmax"| R1
    SOFT         -->|"argmax"| R2
    R1           -->|"score"| CONF
    R2           -->|"score"| CONF

    style INPUTS   fill:#0a1520,stroke:#00e5ff,stroke-width:2px,color:#00e5ff
    style FLASH    fill:#071020,stroke:#00b4cc,stroke-dasharray:4 3,color:#00b4cc
    style HARDWARE fill:#071020,stroke:#00b4cc,stroke-dasharray:4 3,color:#00b4cc
    style RUNTIME  fill:#110a00,stroke:#ffc800,stroke-width:2px,color:#ffc800
    style INFER    fill:#0a150a,stroke:#a8ff3e,stroke-width:2px,color:#a8ff3e
    style OUTPUT   fill:#100a1a,stroke:#8264ff,stroke-width:2px,color:#8264ff

    style M    fill:#071828,stroke:#00e5ff,color:#aaeeff
    style I    fill:#071828,stroke:#00e5ff,color:#aaeeff
    style MCU  fill:#0d2030,stroke:#00e5ff,color:#cceeff

    style OP1  fill:#1a1200,stroke:#ffc800,color:#fff0aa
    style OP2  fill:#1a1200,stroke:#ffc800,color:#fff0aa
    style OP3  fill:#1a1200,stroke:#ffc800,color:#fff0aa

    style INV  fill:#0d1f0d,stroke:#a8ff3e,color:#ccffaa
    style SOFT fill:#0d1f0d,stroke:#a8ff3e,color:#ccffaa

    style R1   fill:#150e22,stroke:#8264ff,color:#d4c8ff
    style R2   fill:#150e22,stroke:#8264ff,color:#d4c8ff
    style CONF fill:#150e22,stroke:#8264ff,color:#d4c8ff
```
---
## 📂 Project Structure
```bash
phase3_task2/
│
├── train.py
├── wafer_classifier_float32.tflite
├── wafer_int8.tflite
├── model_data.h
├── model_labels.h
├── test.png
├── test_image.h
├── img_to_c.py
└── proof_images/
```
📊 Model Optimization Comparison
Model Type	Size	Accuracy	Deployment
Float32	~2.0 MB	High	Desktop
INT8	~694 KB	Slightly Reduced	Embedded MCU

🔄 Complete Workflow Animation Flow

```mermaid
sequenceDiagram
    autonumber

    participant Dev  as 👨‍💻 Developer
    participant Py   as ⚙️ Training Script<br/>(train.py)
    participant TFL  as 🔬 TFLite Converter
    participant MCU  as 🔲 Embedded Device<br/>(i.MX RT1170)

    rect rgb(10, 30, 50)
        Note over Dev,Py: 🏋️ Phase 1 — Model Training
        Dev  ->>+ Py  : Train Model (Keras / TensorFlow)
        Py   -->>- Dev : Float32 Model Weights saved
    end

    rect rgb(20, 15, 5)
        Note over Py,TFL: ⚡ Phase 2 — Quantization & Export
        Dev  ->>+ TFL : Convert model → .tflite
        TFL  ->>  TFL : INT8 Post-Training Quantization
        Note right of TFL: 4× smaller · faster inference
        TFL  -->>- Dev : wafer_int8.tflite ready
    end

    rect rgb(15, 5, 30)
        Note over Dev,TFL: 🔧 Phase 3 — C Header Generation
        Dev  ->>+ TFL : Run xxd / Python script
        TFL  -->>- Dev : model_data.h (uint8_t array)
    end

    rect rgb(5, 20, 5)
        Note over Dev,MCU: 🚀 Phase 4 — Firmware Deployment
        Dev  ->>+ MCU : Flash firmware (arm-none-eabi)
        Note right of MCU: BOARD_FLASH: 2.57% used<br/>Build: 0 errors · 5.1ms
        MCU  ->>  MCU : TFLite Micro — Invoke()
        MCU  -->>- Dev : Classification Output ✅
    end
```

---

## 📊 Model Comparison — Float32 vs INT8

> 📸 **Build verified on:** `evkbmimxrt1170` · NXP i.MX RT1170 · ARM Cortex-M7 · `arm-none-eabi-c++ 14.2.1`

| Attribute | 🔵 Float32 Model | 🟢 INT8 Quantized |
|-----------|:----------------:|:-----------------:|
| **Model Size** | ~2.0 MB | **~694 KB** ✅ |
| **Size Reduction** | baseline | **~65% smaller** 🔽 |
| **Accuracy** | High (reference) | Slightly reduced ≈ −1–2% |
| **Precision** | 32-bit float | 8-bit integer ||
| **Deployment Target** | Desktop / Server | **Embedded MCU** 🔲 |
| **Framework** | TensorFlow | **TFLite Micro** ||

---

## 🏗️ Actual Build Report — `evkbmimxrt1170`

> Captured from MCUXpresso IDE build console · `Build Finished: 0 errors, 1 warning · 5.101ms`

### 🧮 Binary Segment Sizes

```
 Segment     Size        Description
─────────────────────────────────────────────────
 .text       903,216 B   Code + read-only data (Flash)
 .data       819,328 B   Initialized globals (Flash → SRAM)
 .bss         41,648 B   Zero-initialized globals (SRAM)
─────────────────────────────────────────────────
 TOTAL     1,764,192 B   (hex: 0x1aeb60)
```
 <div align="center">
  <img src="proof_images/Screenshot from 2026-02-25 13-07-37.png" width="850"/>
</div>
### 🗃️ Memory Region Utilization

| Memory Region | Used | Total | % Used | Status |
|---------------|-----:|------:|-------:|--------|
| `BOARD_FLASH` | 1,722,544 B ≈ **1.64 MB** | 64 MB | 2.57% | 🟢 OK |
| `BOARD_SDRAM` | 852,784 B ≈ **833 KB** | 48 MB | 1.69% | 🟢 OK |
| `SRAM_DTC_cm7` | 8 KB | 256 KB | 3.12% | 🟢 OK |
| `SRAM_ITC_cm7` | 72 B | 256 KB | 0.03% | 🟢 OK |
| `NCACHE_REGION` | 0 B | 16 MB | 0.00% | ⚪ Unused |
| `SRAM_OC1` | 0 B | 512 KB | 0.00% | ⚪ Unused |
| `SRAM_OC2` | 0 B | 512 KB | 0.00% | ⚪ Unused |
| `SRAM_OC_ECC1` | 0 B | 64 KB | 0.00% | ⚪ Unused |
| `SRAM_OC_ECC2` | 0 B | 64 KB | 0.00% | ⚪ Unused |

### ✅ Build Summary

```
Target  : evkbmimxrt1170_tflm_label_image_cm7.axf
Linker  : arm-none-eabi-c++ (ARM GCC 14.2.1)
Errors  : 0
Warnings: 1
Time    : 5.101 ms
```

---
