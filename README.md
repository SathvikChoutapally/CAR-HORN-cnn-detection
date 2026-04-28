# 🚗 CAR-HORN-CNN-Detection

> **CNN-based car model identification using horn sound analysis** — Mel Spectrogram & MFCC features with GoogLeNet and ResNet50 (PyTorch)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SathvikChoutapally/CAR-HORN-cnn-detection/blob/main/notebooks/ML_PROJECT.ipynb)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project explores whether a deep learning model can **identify a car model purely from its horn sound**. Raw audio recordings of car horns are transformed into visual representations (Mel Spectrograms and MFCCs) and fed into fine-tuned CNN architectures — **GoogLeNet** and **ResNet50** — to classify the car model.

The full pipeline covers:
- Audio collection → chunking → augmentation → spectrogram/MFCC generation
- 70/15/15 train/val/test split
- Transfer learning with GoogLeNet and ResNet50
- Evaluation with accuracy, F1-score, confusion matrix, and Top-3 accuracy
- Single-file inference from a new `.m4a` / `.wav` audio sample

---

## 📁 Repository Structure

```
CAR-HORN-cnn-detection/
│
├── notebooks/
│   └── ML_PROJECT.ipynb          # Full pipeline: preprocessing → training → evaluation
│
├── presentation/
│   └── CarHorn.pptx              # Project presentation slides
│
├── data/
│   └── README.md                 # Dataset download links and structure info
│
└── README.md
```

---

## 🔄 Pipeline

```
Raw Audio (.wav/.m4a)
        │
        ▼
  Audio Chunking (2s chunks, 50% overlap)
        │
        ▼
  Data Augmentation
  (Gaussian Noise, Pitch Shift ±, Time Stretch, Gain, Shift)
        │
        ▼
  Feature Extraction
  ┌─────────────────────┐
  │  Mel Spectrogram    │
  │  MFCC (40 coeffs)   │
  └─────────────────────┘
        │
        ▼
  Train/Val/Test Split  (70% / 15% / 15%)
        │
        ▼
  CNN Fine-tuning
  ┌────────────┬──────────────┐
  │ GoogLeNet  │   ResNet50   │
  └────────────┴──────────────┘
        │
        ▼
  Evaluation & Inference
```

---

## 🧠 Models

### GoogLeNet (Inception v1)
- Pretrained on ImageNet, fine-tuned with frozen backbone
- Final FC layer replaced for `num_classes` output
- Trained with auxiliary classifiers enabled

### ResNet50
- Pretrained on ImageNet
- Final FC replaced with `Dropout(0.5) → Linear(num_classes)`
- Used for inference with confidence threshold + entropy-based rejection

---

## 📊 Dataset

### Raw Audio Dataset (Total Dataset)
Contains horn sound recordings of various random car models.

📥 **[Download Total Dataset (Google Drive)](https://drive.google.com/drive/folders/1wIjJkNlkF47JE0VTlO8Cvfe8SVdVRLC1?usp=drive_link)**

### Preprocessed Split Dataset (Train / Val / Test)
Pre-split dataset containing Mel Spectrogram and MFCC images, ready for model training.

📥 **[Download Split Dataset (Google Drive)](https://drive.google.com/drive/folders/11Yii8IWjF6t0zIyM5xfv00euW_XGN8i_?usp=drive_link)**

#### Dataset Structure (after download)
```
dataset/
├── train/
│   ├── CarModel_A/
│   │   ├── chunk000_orig_spec.png
│   │   ├── chunk000_orig_mfcc.png
│   │   ├── chunk000_noise_spec.png
│   │   └── ...
│   └── CarModel_B/ ...
│
├── val/
│   └── ... (same structure)
│
└── test/
    └── ... (same structure)
```

> **Note:** The dataset is not included in this repository due to its large size. Download from the links above and place in `/content/drive/MyDrive/dataset/` if using Google Colab.

---

## ⚙️ Setup & Installation

### Requirements
```bash
pip install torch torchvision librosa audiomentations matplotlib numpy pydub scikit-learn seaborn
apt-get install -y ffmpeg
```

### Run in Google Colab (Recommended)
1. Click the **"Open in Colab"** badge at the top
2. Mount your Google Drive
3. Download the dataset from the links above into your Drive
4. Run all cells in order

---

## 🚀 Notebook Walkthrough

The notebook `ML_PROJECT.ipynb` is organized into 5 sections:

| Section | Description |
|---------|-------------|
| **1. Augmentation & Fragmentation** | Chunks raw audio → applies 7 augmentations → generates Mel Spectrograms & MFCC images |
| **2. Data Splitting** | Splits processed image dataset into 70% train / 15% val / 15% test |
| **3. GoogLeNet Model** | Loads pretrained GoogLeNet, fine-tunes, trains with GPU support |
| **4. ResNet50 Model** | Loads pretrained ResNet50, fine-tunes, trains with early stopping |
| **5. Inference & Results** | Single audio prediction with confidence threshold + confusion matrix + classification report |

---

## 🔊 Inference (Predicting a New Horn)

```python
predict_horn("/path/to/your_horn.m4a", threshold=90)
```

**Output example:**
```
🎵 Audio: your_horn.m4a
🔝 Top Predictions:
  Toyota_Innova   → 94.31%
  Honda_City      → 3.12%
  Maruti_Swift    → 1.43%

🚗 Final Prediction: Toyota_Innova
📊 Confidence: 94.31%
```

The model uses two rejection mechanisms:
- **Confidence threshold** — rejects if top prediction < 90%
- **Entropy check** — rejects if prediction distribution is too uncertain

---

## 📈 Evaluation Metrics

The results notebook computes:
- Top-1 and Top-3 Accuracy
- Precision, Recall, F1-score (macro & weighted)
- Per-class Classification Report
- Confusion Matrix (heatmap)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training & inference |
| Torchvision | Pretrained GoogLeNet & ResNet50 |
| Librosa | Audio loading, Mel Spectrogram, MFCC |
| Audiomentations | Audio augmentation |
| Scikit-learn | Metrics & confusion matrix |
| Seaborn / Matplotlib | Visualization |
| Google Colab | Training environment (GPU) |

---

## 📂 How to Add Dataset to Your Own Run

1. Download the dataset from the Google Drive links above
2. Upload to your Google Drive under `MyDrive/dataset/`
3. In the notebook, the paths are already configured:
   ```python
   train_dir = "/content/drive/MyDrive/dataset/train"
   val_dir   = "/content/drive/MyDrive/dataset/val"
   test_dir  = "/content/drive/MyDrive/dataset/test"
   ```

---

## 🙋 Author

**Sathvik Choutapally**
- GitHub: [@SathvikChoutapally](https://github.com/SathvikChoutapally)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
