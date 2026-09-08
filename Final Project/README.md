# Computer Vision — Final Project: Deep Image Restoration & Inpainting via U-Net

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15%2B-orange.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-yellow.svg)](https://gradio.app/)
[![Course](https://img.shields.io/badge/Course-Computer_Vision_Spring_2026-purple.svg)](#)

> **Author:** Nazhin Nikkhahbahrami  
> **Student ID:** 810102530  
> **Course:** Computer Vision (Spring 2026)

---

## 📌 Overview

This repository contains the complete implementation, training pipeline, quantitative benchmarking, and interactive deployment for the **Final Project** of the Computer Vision course.

The project addresses **deep image restoration and inpainting** on the `openphoto-restore-dataset`. The objective is to reconstruct old, physically degraded photographs corrupted by dust, speckle noise, deep scratches, and localized tears back into pristine, high-fidelity images. Using a customized **U-Net architecture with lateral skip connections** implemented in PyTorch, the system achieves a **PSNR of 22.73 dB** (exceeding the $\ge 20 \text{ dB}$ threshold) and an **SSIM of 0.8476** (exceeding the $\ge 0.65$ threshold). The project also features an interactive web interface powered by **Gradio** for real-time user testing.

---

## 📁 Repository Structure

```text
Final Project/
├── CV-FinalProject-810102530.ipynb  # Complete PyTorch pipeline, training logs, plots & UI
├── CV-Final Project.pdf             # Project problem statement & technical specifications
└── README.md                        # Project documentation
```

> **Dataset Source:** Hugging Face [`openphoto-restore-dataset`](https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset), cloned locally via `git lfs` for reliable batch streaming.

---

## 🔬 Project Architecture & Pipeline Breakdown

### 1. Data Preparation & Preprocessing Pipeline

1. **Paired Image Dataset Ingestion:**
   - Ingests paired samples of physically damaged images ($X_{\text{damaged}}$) and corresponding clean ground-truth originals ($Y_{\text{clean}}$).
   - Cloned locally using `git lfs` to prevent network timeouts during large-scale tensor transfers.
2. **Transforms & Partitioning:**
   - Standardized all spatial resolutions to $256 \times 256$ pixels.
   - Converted to PyTorch tensors with floating-point normalization in the range $[0.0, 1.0]$.
   - Partitioned the primary training set into **90% Training** and **10% Validation**, reserving the provided test set strictly for unseen final evaluation.
   - Batched with PyTorch `DataLoader` (batch size $= 16$, training set shuffled).

---

### 2. Model Selection & Architecture (U-Net)

Traditional Convolutional Autoencoders suffer severe spatial detail loss through bottleneck downsampling, resulting in overly blurred reconstructions. The **U-Net** architecture was selected to preserve high-frequency spatial cues via multi-scale skip connections.

```text
Input (3 x 256 x 256)
  │
  ├──> DoubleConv(64)  ───[ Skip Connection ]───────────────────┐
  │        ↓ MaxPool(2)                                         │
  ├──> DoubleConv(128) ───[ Skip Connection ]────────────┐      │
  │        ↓ MaxPool(2)                                  │      │
  ├──> DoubleConv(256) ───[ Skip Connection ]─────┐      │      │
  │        ↓ MaxPool(2)                           │      │      │
  ├──> DoubleConv(512) ───[ Skip Connection ]──┐  │      │      │
  │        ↓ MaxPool(2)                        │  │      │      │
  └──> DoubleConv(1024) [Bottleneck]           │  │      │      │
           ↓ ConvTranspose2d(512)              │  │      │      │
       Concat + DoubleConv(512) <──────────────┘  │      │      │
           ↓ ConvTranspose2d(256)                 │      │      │
       Concat + DoubleConv(256) <─────────────────┘      │      │
           ↓ ConvTranspose2d(128)                        │      │
       Concat + DoubleConv(128) <────────────────────────┘      │
           ↓ ConvTranspose2d(64)                                │
       Concat + DoubleConv(64)  <───────────────────────────────┘
           ↓ Conv2d(3, 1x1) + Sigmoid
Output (3 x 256 x 256)
```

- **DoubleConv Blocks:** Consecutive $3 \times 3$ convolutions each followed by `BatchNorm2d` and `ReLU` activation.
- **Contracting Path (Encoder):** 4 downsampling stages with $2 \times 2$ max pooling ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$).
- **Bottleneck:** Deepest layer with 1024 channels holding abstract semantic context.
- **Expansive Path (Decoder):** 4 upsampling stages via `ConvTranspose2d` concatenated with matching encoder feature maps to recover fine textures and sharp boundaries.
- **Output Activation:** $1 \times 1$ convolution mapped through `Sigmoid` to guarantee normalized pixel outputs in $[0.0, 1.0]$.

---

### 3. Model Training & Convergence Dynamics

- **Loss Function:** Mean Squared Error (**MSE Loss**) to directly optimize per-pixel fidelity and maximize PSNR.
- **Optimizer:** Adam ($\text{lr} = 10^{-4}$ / $0.0001$).
- **Hyperparameters:** Batch size $= 16$, Total Epochs $= 15$, Resolution $= 256 \times 256$.
- **Automated Checkpointing:** Integrated checkpoint saving (`training_checkpoint.pth`) tracking `best_model_weights` and `best_val_loss`.
- **Training Analysis:** Both training and validation loss converged smoothly across 15 epochs, reaching a final validation loss of **0.007876** without divergence or overfitting.

---

### 4. Quantitative Results & Evaluation Benchmarks

The model was evaluated across the unseen test set using Peak Signal-to-Noise Ratio (**PSNR**) and Structural Similarity Index Measure (**SSIM**):

| Metric | Achieved Value | Required Threshold | Performance Margin | Status |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR** | **22.73 dB** | $\ge 20.0 \text{ dB}$ | **+2.73 dB** | ✅ **Passed** |
| **SSIM** | **0.8476** | $\ge 0.6500$ | **+0.1976** | ✅ **Passed** |

> **Epoch Stopping Rationale:** Halting training at 15 epochs avoided the "PSNR-chasing" trap where models minimize pixel MSE by producing overly smoothed averages. Stopping at this equilibrium point maximized perceptual and structural similarity (**SSIM 0.8476**) while comfortably exceeding the PSNR target.

---

### 5. Qualitative Analysis, Limitations & Future Improvements

1. **Qualitative Strengths:**
   - Near-perfect excision of thin-to-medium physical scratches, dust specks, and grain noise.
   - Sharp boundary preservation on natural objects, human figures, and architectural edges.
2. **Failure Modes & Inherent MSE Limitations:**
   - **Regression-to-the-Mean:** Because MSE heavily penalizes wrong color guesses quadratically, the model plays mathematically "safe" on missing chromatic data, resulting in muted or desaturated grayish tones on vibrant scenes (e.g., orange sunsets or vivid red clothing).
   - **Thick Tear Inpainting:** Extremely wide, opaque scratches are filled with plausible smooth textures but leave slight softening artifacts.
3. **Future Roadmap (Transition to GANs):**
   - Upgrading the framework to a **Conditional Generative Adversarial Network (cGAN / Pix2Pix)** using the U-Net as the Generator paired with a PatchGAN Discriminator.
   - Adding **Perceptual Loss** (VGG-based feature matching) to enforce realistic texture synthesis and chromatic vividness beyond pixel-wise $L_2$ distance.

---

### 6. Interactive Deployment (Gradio Web UI)

An interactive browser UI was implemented using **Gradio** (`gr.Interface`) allowing instant testing:
- **Interactive Drag-and-Drop:** Upload any real-world degraded photo.
- **Automated Pipeline:** Resizes to $256 \times 256$, normalizes, executes GPU inference through U-Net, and renders side-by-side restoration in real time.

---

## 🚀 How to Run

### 1. Prerequisites
Install PyTorch (CUDA recommended), Gradio, and required libraries:

```bash
pip install torch torchvision datasets scikit-image gradio pillow tqdm matplotlib
```

### 2. Running the Notebook & Launching the Web UI
Open the notebook in Jupyter Lab, Kaggle, or Google Colab:

```bash
cd "Final Project"
jupyter notebook CV-FinalProject-810102530.ipynb
```

1. Run the preprocessing and model training cells (or load the saved checkpoint `training_checkpoint.pth`).
2. Run the evaluation cell to calculate the PSNR/SSIM metrics on the test set.
3. Launch the final code cell to start the interactive **Gradio** web interface.
