# Distorted Visual Sequence Recognition (CRNN + CTC)

A complete deep learning pipeline for recognizing text sequences from heavily distorted grayscale images (such as CAPTCHAs or noisy/skewed visual sequences) using a **Convolutional Recurrent Neural Network (CRNN)** and **Connectionist Temporal Classification (CTC)** loss.

This repository contains a standalone command-line inference script (`predict.py`) to run predictions on single image files using the trained model weights.

---

##  Quick Start (Running Inference on Single Images)

If you have a trained weights file (`best_crnn_model.pth`) and want to predict the text in a custom image, follow these steps:

### 1. Requirements & Setup
* Place the pre-trained weights file `best_crnn_model.pth` in the root folder of this project.
* **Download Weights**: https://drive.google.com/file/d/1BYCitVPpN9OI8pmS-Ppuy6wGPw2KUqw2/view?usp=drive_link

### 2. Run Inference
You can run the prediction script using the local embedded Python environment (which contains all the required libraries like PyTorch, Pillow, OpenCV, and Pandas):

```powershell
.\python-embed\python.exe predict.py <path_to_your_image.png>
```

*Example:*
```powershell
.\python-embed\python.exe predict.py test-0.png
```

If you are using a standard global Python installation with PyTorch already installed:
```bash
python predict.py <path_to_your_image.png>
```

---

## 🧠 Model Architecture

The model uses a classic **CRNN + CTC** architecture standard for text sequence recognition:

```
Input Image (H×W grayscale, automatically resized to 32×128)
    │
    ▼
[CNN Backbone (VGG-inspired)]  ──► Extracts local visual features (H is collapsed to 1)
    │
    ▼  (sequence of feature columns: [W/4, Batch, 512])
[BiLSTM (2 layers, 256 hidden)] ──► Captures sequential context (left-to-right & right-to-left)
    │
    ▼  (per-timestep logits: [W/4, Batch, 40])
[CTC Decoder (Greedy)]         ──► Collapses consecutive repeats and removes blank tokens
    │
    ▼
Predicted Text String
```

### Key Technical Specs:
* **Input Constraints**: Automatically resizes any custom image dimension to **32px (Height) × 128px (Width)** before running inference.
* **Character Vocabulary**: Configured for 40 classes:
  `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.` (along with the CTC blank token at index 0).

---

## 🛠️ Folder Structure

```
├── predict.py                  # Standalone CLI inference script
├── best_crnn_model.pth         # Trained weights checkpoint file
└── README.md                   # This project documentation file
```
