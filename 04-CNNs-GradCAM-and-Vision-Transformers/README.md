# 👁️ CNNs, Grad-CAM & Vision Transformers (ViT)

This module explores modern computer vision paradigms, spanning classical deep CNN feature representations, gradient-weighted class activation mapping (Grad-CAM) for network interpretability, and Vision Transformers (ViT).

---

## 📌 Module Breakdown

### 🔬 01. Deep CNN Architectures & Feature Extraction
- **Directory**: `01-CNN-Architectures/`
- **Notebook**: `01-CNN-Architectures/HW3_Q1_CNN_Architectures.ipynb`
- **Topics**: Convolutional layers, pooling mechanisms, receptive fields, residual connections, and feature hierarchy analysis on standard image classification benchmarks.

### 🔍 02. Network Interpretability & Grad-CAM Visualization
- **Directory**: `02-GradCAM-Model-Interpretability/`
- **Notebook**: `02-GradCAM-Model-Interpretability/HW3_Q2_GradCAM_Interpretability.ipynb`
- **Topics**: Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize what regions deep networks attend to when making predictions.
- **Outputs**:
  - `outputs_sec12/` & `outputs_sec24/`: High-resolution layer-by-layer attention heatmaps across intermediate layers (layer 0, 10, 15, 20, 30, 35, 40).
  - Demonstrates how semantic abstraction evolves from low-level edges to high-level object concepts.

### 🚀 03. Vision Transformers (ViT)
- **Directory**: `03-Vision-Transformers-ViT/`
- **Notebook**: `03-Vision-Transformers-ViT/HW3_Q3_Vision_Transformers.ipynb`
- **Topics**: Patch tokenization, positional embeddings, multi-head self-attention (MSA), and transformer encoder blocks applied directly to image classification without inductive convolutional bias.

---

## 📄 Course Submission Report
Detailed mathematical analysis and empirical experimental results are documented in:
- `HW3-Bagherinejad-402200359.pdf`
