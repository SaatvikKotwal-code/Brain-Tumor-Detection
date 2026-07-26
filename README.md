# 🩺 MediScan AI — Medical AI Assistant & Brain MRI Classifier

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Vision Transformer](https://img.shields.io/badge/ViT-95.19%25%20Accuracy-00599C?style=for-the-badge&logo=python&logoColor=white)](https://github.com/google-research/vision_transformer)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**MediScan AI** is an advanced healthcare assistant and diagnostic visualization platform powered by Deep Learning. It combines a **Vision Transformer (ViT)** neural network for automated **Brain MRI Scan Classification** with a comprehensive suite of interactive health tools: **Symptom Checker**, **Medication Directory**, **Emergency First Aid Protocols**, and **Health & Wellness Recommendations**.

---

## 🌟 Key Features

### 🧠 1. Brain MRI AI Classifier (Vision Transformer)
- **High-Accuracy Deep Learning Model**: Fine-tuned PyTorch ViT (`vit_tiny_patch16_224` & custom Multi-Head Self-Attention ViT architecture) trained on brain MRI scans.
- **4-Class Diagnostic Categories**:
  - 🔴 **Glioma Tumor**
  - 🟠 **Meningioma Tumor**
  - 🟢 **No Tumor Detected** (Healthy brain tissue)
  - 🟣 **Pituitary Tumor**
- **High Performance Benchmark**: Achieves **95.19% Validation Accuracy**.
- **Interactive ROI Heatmap**: Canvas-based heatmap visualizer highlighting Region of Interest (ROI) regions on uploaded brain scan slices.
- **Flexible Image Upload**: Supports PNG, JPG, and export formats with pre-loaded sample MRI scans for instant browser testing.

### 🩺 2. Interactive Symptom Checker
- Guided, multi-step symptom diagnostic wizard.
- Calculates likelihood scores matching user symptoms against verified clinical condition profiles.
- Displays triage urgency levels: **Immediate Care**, **Urgent Care**, **Routine Consultation**, and **Self-Care**.

### 💊 3. Medication Directory & Search
- Searchable clinical drug database (`data/medications.json`).
- Detailed medication cards containing indications, standard dosage guidelines, contraindications, and potential side effects.

### 🚑 4. Emergency First Aid Guides
- Actionable, step-by-step emergency response protocols (`data/first-aid.json`).
- Clear guidelines for emergency scenarios: CPR, Choking, Severe Bleeding, Burns, Fractures, Poisoning, and Heatstroke.

### 💡 5. Health & Wellness Engine
- Evidence-based preventive healthcare advice categorized into Nutrition, Physical Fitness, Mental Health, Sleep Hygiene, and Cardiovascular Health (`data/health-tips.json`).

---

## 🏗 Architecture & Tech Stack

```
                                ┌─────────────────────────────────────────┐
                                │             MediScan AI Web             │
                                │        (Vanilla HTML5 / CSS3 / JS)      │
                                └────────────────────┬────────────────────┘
                                                     │
                                       HTTP / REST API (JSON / FormData)
                                                     │
                                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       Flask Python Backend                                         │
 │                                           (server.py)                                              │
 └──────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   Vision Transformer Engine                                       │
 │                      (best_vit_brain_mri.pth | PyTorch CUDA / CPU Acceleration)                    │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Technologies & Frameworks |
| :--- | :--- |
| **Frontend UI** | HTML5, Modern Vanilla CSS3 (Design Tokens, Glassmorphism, CSS Grid/Flexbox), ES6+ JavaScript |
| **Backend API** | Python 3.10+, Flask, Flask-CORS |
| **Deep Learning** | PyTorch, `timm` (PyTorch Image Models), Torchvision, Pillow |
| **Model Architecture** | Vision Transformer (`vit_tiny_patch16_224` / Custom Patch Embedding + Multi-Head Self-Attention) |
| **Datasets** | Pre-categorized JSON datasets (`data/symptoms.json`, `data/medications.json`, `data/first-aid.json`, `data/health-tips.json`) |

---

## 📁 Repository Structure

```
CrickEye_Project/
├── best_vit_brain_mri.pth   # Pre-trained ViT Model Checkpoint (95.19% Val Accuracy)
├── train_vit.py             # ViT Training & Data Augmentation Pipeline
├── vit_model.py             # Custom ViT Architecture (PatchEmbedding, MHSA, TransformerBlock)
├── server.py                # Flask REST API Backend & Static Web Server
├── index.html               # Main SPA Web Application Interface
├── css/
│   ├── animations.css       # Keyframe animations & UI state transitions
│   ├── components.css       # UI Components (Cards, Badges, Modals, Buttons)
│   ├── index.css            # Base styles, CSS Variables, Typography & Reset
│   └── pages.css            # Section-specific views (Analyzer, Symptom Checker, etc.)
├── js/
│   ├── app.js               # Application Router & Navigation Controller
│   ├── image-analyzer.js    # MRI Upload Handler, API Communication & Heatmap Overlay
│   ├── symptom-checker.js   # Symptom Quiz Engine & Triage Matcher
│   ├── medication-db.js     # Drug Search & Filter Engine
│   ├── first-aid.js         # Emergency Guide Renderer
│   ├── health-tips.js       # Health & Wellness Content Controller
│   └── utils.js             # Shared DOM Utilities & Toast Notification Helpers
└── data/
    ├── first-aid.json       # Emergency First Aid Protocols Dataset
    ├── health-tips.json     # Wellness & Preventive Tips Dataset
    ├── medications.json     # Clinical Medication Information Dataset
    └── symptoms.json        # Symptom Diagnostics Dataset
```

---

## ⚙️ Quickstart & Setup

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **NVIDIA GPU with CUDA** (Optional, falls back automatically to CPU)

### 2. Install Dependencies
Clone the repository and install required packages:

```bash
# Clone repository
git clone https://github.com/your-username/MediScan-AI.git
cd MediScan-AI

# Install required dependencies
pip install torch torchvision timm flask flask-cors pillow
```

### 3. Launch the Server
Start `server.py` to run the Flask API and load the ViT model checkpoint:

```bash
python server.py
```

Expected output:
```text
Loading trained ViT model from 'best_vit_brain_mri.pth' on cuda...
ViT Model successfully loaded! Best Val Accuracy: 95.19%
Starting MediScan AI PyTorch ViT Backend on http://localhost:5000
```

### 4. Open in Browser
Open your browser and go to:
👉 **`http://localhost:5000`**

---

## 📡 REST API Reference

### 1. Backend Status (`GET /api/status`)
Checks server availability and model details.

**Response Example:**
```json
{
  "status": "online",
  "model_name": "vit_tiny_patch16_224",
  "device": "cuda",
  "val_accuracy": 95.19,
  "classes": ["glioma", "meningioma", "notumor", "pituitary"],
  "is_loaded": true
}
```

### 2. Brain MRI Classification (`POST /api/predict`)
Classifies an uploaded MRI image. Accepts `multipart/form-data` with key `image` or JSON with base64 payload.

**Example Request:**
```bash
curl -X POST -F "image=@sample_brain_mri.jpg" http://localhost:5000/api/predict
```

**Response Example:**
```json
{
  "success": true,
  "top_prediction": {
    "id": "notumor",
    "name": "No Tumor Detected",
    "color": "#10b981",
    "desc": "Healthy brain tissue scan showing no signs of abnormal tumorous growth.",
    "probability": 0.9842
  },
  "predictions": [
    { "id": "notumor", "name": "No Tumor Detected", "probability": 0.9842 },
    { "id": "glioma", "name": "Glioma Tumor", "probability": 0.0091 },
    { "id": "meningioma", "name": "Meningioma Tumor", "probability": 0.0043 },
    { "id": "pituitary", "name": "Pituitary Tumor", "probability": 0.0024 }
  ],
  "model": "Vision Transformer (ViT)",
  "accuracy": "95.19%"
}
```

---

## 🎯 Model Training & Fine-Tuning

To train or fine-tune the Vision Transformer model on a custom Brain MRI dataset:

### Dataset Directory Format
Organize your dataset inside `./data`:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Run Training Script
```bash
python train_vit.py --data_dir ./data --epochs 30 --batch_size 16 --lr 1e-4 --use_timm
```

**Training Parameters:**
- `--data_dir`: Root dataset path (default: `./data`)
- `--epochs`: Total training epochs (default: `30`)
- `--batch_size`: Batch size per step (default: `16`)
- `--lr`: Learning rate (default: `0.0001`)
- `--use_timm`: Flag to build via `timm.create_model('vit_tiny_patch16_224')` or custom ViT architecture

The trained model checkpoint will automatically be saved to `best_vit_brain_mri.pth`.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: **MediScan AI** is strictly an educational and research project. It is **NOT** a clinical diagnostic tool or medical device and does **NOT** provide medical diagnoses or professional healthcare advice. Always consult a licensed healthcare professional for any medical evaluation or diagnosis.

---

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.
