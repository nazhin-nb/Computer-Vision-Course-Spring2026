# Computer Vision — Homework 4: Natural Scene Classification with Deep CNNs

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange.svg)](https://numpy.org/)
[![Course](https://img.shields.io/badge/Course-Computer_Vision_Spring_2026-purple.svg)](#)

> **Author:** Nazhin Nikkhahbahrami  
> **Student ID:** 810102530  
> **Course:** Computer Vision (Spring 2026)

---

## 📌 Overview

This repository contains the end-to-end design, implementation, and optimization of a deep **Convolutional Neural Network (CNN)** for multi-class natural scene image classification in **Homework 4** of the Computer Vision course. 

The objective is to classify landscape and urban scenes into 6 distinct categories (**Buildings, Forest, Glacier, Mountain, Sea, Street**). The work encompasses data loading, pipeline augmentation, architectural parameter analysis, diagnostic learning curve evaluation, detailed confusion matrix error analysis, and systematic regularization strategies (Dropout, learning rate reduction, and channel expansion) boosting test accuracy from **81.87%** to **84.53%**.

---

## 📁 Repository Structure

```text
HW4/
├── CV-HW4-810102530.ipynb   # Complete Jupyter Notebook with models, training logs & curves
├── CV-HW4.pdf               # Homework assignment specifications
└── README.md                # Project documentation
```

> **Dataset:** Uses the Intel Image Classification benchmark dataset partitioned into `seg_train` (80% train, 20% validation) and `seg_test` subsets.

---

## 🔬 Homework Breakdown

### Part 1: Dataset Exploration & Architecture Fundamentals

1. **Data Ingestion & Batching:**
   - Ingested images with `tf.keras.utils.image_dataset_from_directory` with an 80/20 train/validation split and batch size of 32.
   - Standardized spatial resolutions to $150 \times 150 \times 3$ RGB.
2. **Role of the `Flatten` Layer:**
   - Convolutional and pooling operations preserve spatial topology and produce 3D tensor feature maps ($H \times W \times C$).
   - The `Flatten` layer acts as the critical structural interface, unrolling multidimensional spatial arrays into a continuous 1D feature vector required by dense classification layers.

---

### Part 2: Preprocessing & Data Augmentation

1. **Pixel Normalization:**
   - Scaled raw $[0, 255]$ pixel values into $[0, 1]$ using `Rescaling(1./255)`, ensuring stable numerical gradients and accelerated convergence.
2. **On-the-Fly Data Augmentation:**
   - Integrated a sequence of non-destructive spatial transformations into the model graph:
     - `RandomFlip("horizontal")`: Mirror reflection invariance.
     - `RandomRotation(0.1)`: Slight affine rotation invariance ($\pm 10\%$).
     - `RandomZoom(0.1)`: Scale invariance ($\pm 10\%$).
   - **Overfitting Mitigation:** By generating stochastically perturbed variations during each training epoch, the network is discouraged from memorizing fixed pixel grids and forced to learn robust structural invariants.

---

### Part 3 & 4: Baseline CNN Architecture & Loss Formulation

1. **Layer Hierarchy:**
   - **Feature Extractor:** 4 sequential blocks of `Conv2D` ($3\times3$ kernels, ReLU activation) coupled with `MaxPooling2D` ($2\times2$ spatial subsampling):
     - Block 1: 32 filters
     - Block 2: 64 filters
     - Block 3: 128 filters
     - Block 4: 128 filters
   - **Classifier Head:** `Flatten` $\rightarrow$ `Dense(512, ReLU)` $\rightarrow$ `Dense(6, Softmax)`.
2. **Parameter Distribution Analysis:**
   - Convolutional layers share weight kernels across spatial dimensions, requiring relatively few parameters.
   - The majority of trainable parameters reside in the dense bottleneck transition following the `Flatten` layer.
3. **Loss & Optimizer:**
   - **Loss:** `SparseCategoricalCrossentropy` chosen to handle integer-encoded labels directly without the memory overhead of one-hot expansion.
   - **Optimizer:** Adam with default learning rate ($10^{-3}$).

---

### Part 5, 6 & 7: Baseline Training, Evaluation & Diagnostic Curves

1. **Performance Metrics:**
   - Achieved a baseline test accuracy of **81.87%** with a test loss of **0.6231**, surpassing the 80% course threshold.
   - Best per-class performance: **Forest** ($F_1\text{-score} = 0.94$) and **Street** ($F_1\text{-score} = 0.84$).
2. **Confusion Matrix & Systematic Error Analysis:**
   - **Glacier vs. Mountain:** The primary source of misclassification. Both classes share snowy textures, rugged silhouettes, and cool white-blue palettes.
   - **Buildings vs. Street:** Logical urban co-occurrence and shared geometric lines/man-made textures.
   - **Mountain vs. Sea:** Distant atmospheric haze causes mountains to appear ocean-blue; extensive clear skies mimic open water.
3. **Learning Curve Diagnosis:**
   - **Epochs 1–10:** Healthy simultaneous improvement in both training and validation accuracy/loss.
   - **Epochs 10–20:** Validation metrics plateaued around 81–83% while training accuracy climbed to ~88%, signaling the onset of mild overfitting.

---

### Part 8: Model Optimization & Regularization

To resolve the generalization gap and elevate test accuracy, three targeted architectural enhancements were introduced:

1. **Dropout Regularization (`Dropout(0.3)`):**
   - Inserted before the dense classifier to randomly deactivate 30% of feature activations during training passes, preventing neuron co-adaptation.
2. **Learning Rate Refinement ($\text{lr} = 0.0005$):**
   - Reduced the Adam learning rate by 50% to prevent overshooting sharp local minima and achieve smoother convergence.
3. **Capacity Scaling (Doubled Feature Filters):**
   - Expanded Conv2D filter dimensions to **`[64, 128, 256, 256]`**, substantially increasing representational capacity for fine spatial patterns.

---

## 📊 Performance Benchmark & Comparison

| Model Architecture | Modifications | Test Loss | Test Accuracy | Overfitting Behavior |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline Model** | 4 Conv blocks `[32, 64, 128, 128]`, Adam ($\alpha=10^{-3}$) | 0.6231 | **81.87%** | Mild overfitting past epoch 10 |
| **Intermediate Model** | Added `Dropout(0.3)` + Lower LR ($\alpha=5\times 10^{-4}$) | 0.5219 | **83.30%** | Stable validation loss curve |
| **Final Enhanced Model** | `Dropout(0.3)` + Lower LR + Doubled Filters `[64, 128, 256, 256]` | **0.4750** | **84.53%** | **Optimal convergence & tight generalization gap** |

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Python 3.8+ and the following deep learning packages are installed:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 2. Running the Notebook
Open the notebook in Jupyter Lab or Google Colab:

```bash
cd HW4
jupyter notebook CV-HW4-810102530.ipynb
```

Execute all notebook cells sequentially to:
1. Load and batch the Intel natural scene dataset.
2. Build, train, and evaluate the baseline 4-block CNN.
3. Generate classification reports, confusion matrices, and misclassified image grids.
4. Train the regularized, wide-capacity model and inspect the improved learning trajectories.
