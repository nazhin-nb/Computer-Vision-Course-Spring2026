# Computer Vision — Homework 3: Face Detection, Demographic DNN Inference & Deep Image Colorization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange.svg)](https://numpy.org/)
[![Course](https://img.shields.io/badge/Course-Computer_Vision_Spring_2026-purple.svg)](#)

> **Author:** Nazhin Nikkhahbahrami  
> **Student ID:** 810102530  
> **Course:** Computer Vision (Spring 2026)

---

## 📌 Overview

This repository contains the complete implementation, experimental evaluations, and theoretical analyses for **Homework 3** of the Computer Vision course. The project spans three distinct vision domains:
1. **Classical Object Detection:** Cat face detection using Haar Feature-based Cascade Classifiers.
2. **Deep Demographic Inference:** Multi-stage Deep Neural Network (DNN) pipeline for face localization (Single Shot Detector - SSD with ResNet-10), age estimation, and gender classification, deployed on static portraits and real-time video streams.
3. **Deep Image Colorization:** Self-supervised Convolutional Autoencoder predicting chrominance ($a, b$) channels from grayscale luminance ($L$) in the **CIE $L^*a^*b^*$** color space.

A major focus of this work is **robustness and perturbation stress-testing**, quantitatively and qualitatively assessing model behavior under affine rotations, extreme lighting perturbations, aggressive Gaussian blur, and synthetic noise.

---

## 📁 Repository Structure

```text
HW3/
├── main.ipynb               # Comprehensive Jupyter Notebook containing all experiments & outputs
├── CV_HW3.pdf               # Homework assignment specifications
├── Jobs_2_output.mp4        # Processed video output with real-time face/age/gender inference
└── README.md                # Project documentation
```

> **Note on Pre-trained Models:** The pipeline utilizes pre-trained weights for OpenCV DNN (`opencv_face_detector_uint8.pb`, `age_net.caffemodel`, `gender_net.caffemodel`), OpenCV Haar Cascade (`haarcascade_frontalcatface_extended.xml`), and a trained Keras Autoencoder (`colorize_autoencoder_first1000.keras`).

---

## 🔬 Homework Breakdown

### Part 1: Cat Face Detection with Haar Cascades

1. **Cascade Pipeline Setup:**
   - Employs `haarcascade_frontalcatface_extended.xml` via `cv2.CascadeClassifier`.
   - Applied grayscale conversion and Gaussian smoothing as pre-filtering to suppress high-frequency image noise.
   - Evaluated baseline detection parameters: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(50, 50)`.
2. **False Positive & Error Analysis:**
   - Identified a False Positive in sample `cat1.webp` where an extraneous bounding box locked onto the nose region. Rectangular Haar features evaluate adjacent contrast differentials (light-dark transitions); local nasal shadows mimic the ocular-nasal intensity profile.
3. **Color vs. Grayscale Invariance:**
   - Verified that Haar features evaluate integral images based purely on scalar intensity differentials. Color channels offer no additional discriminative power or false-positive reduction.
4. **Rotational Sensitivity (90° Invariance Test):**
   - Rotated inputs 90° clockwise. The classifier completely failed to detect cat faces.
   - **Root Cause:** Haar features rely strictly on upright, axis-aligned rectangular kernel topologies (e.g., bilateral eye alignment above the muzzle). Breaking the horizontal coordinate alignment destroys feature responses.

---

### Part 2: Human Face, Age & Gender Estimation Pipeline

1. **Multi-Stage DNN Architecture:**
   - **Face Localization:** Single Shot MultiBox Detector (SSD) with ResNet-10 backbone (`cv2.dnn.readNetFromTensorflow`). Generates 300×300 input blobs with mean subtraction `(104.0, 117.0, 123.0)` and confidence thresholding (`conf_threshold=0.7`).
   - **Demographic Classification:** Two Caffe CNNs (`age_net.caffemodel`, `gender_net.caffemodel`) processing 227×227 cropped face patches with 20px contextual boundary padding and mean normalization `(78.4, 87.8, 114.9)`.
   - **Target Labels:**
     - **Age Brackets (8 classes):** `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`.
     - **Gender (2 classes):** `Male`, `Female`.
2. **Robustness & Perturbation Stress Testing:**
   - **90° Geometric Rotation:** Standard CNNs lack inherent rotation invariance; faces failed detection across 3 of 4 test subjects. Only `musk.jpg` succeeded due to high-contrast studio illumination and clean isolated background.
   - **Illumination Shifts:**
     - **Darkened ($\times 0.5$):** Severe shadow crush caused face detection failure on low-contrast, pre-shadowed subjects (`man.jpg`).
     - **Brightened ($\times 1.8$):** Successful localization, but highlight saturation washed out subtle facial creases, slightly biasing age predictions younger.
   - **Aggressive Gaussian Blurring ($51 \times 51$ Kernel):**
     - Evaluated robustness under catastrophic loss of high spatial frequencies.
     - **Age Failure:** Elderly subject (`man.jpg`) was misclassified as `(8-12)` years old because age classifiers heavily rely on high-frequency micro-textures (wrinkles, skin pores, and eye creases).
     - **Gender Confusion:** Extreme facial smoothing caused masculine jawline and brow ridge cues to blur, leading to gender inversion (`musk.jpg` classified as `Female`).
3. **Continuous Video Stream Inference (`Jobs_2_output.mp4`):**
   - Deployed pipeline on a continuous video segment of Steve Jobs (`Jobs_2 .mp4`).
   - **Temporal Fluctuation Analysis:** Frame-by-frame age estimations fluctuated between `(38-43)` and younger brackets. Specular reflections from dynamic stage lighting created transient hot-spots on the forehead/cheeks, mimicking the smoothing effect of the blur test.

---

### Part 3: Deep Image Colorization (Convolutional Autoencoder)

1. **Autoencoder Architecture in CIE $L^*a^*b^*$ Space:**
   - Instead of predicting full RGB tuples, the model utilizes the **CIE $L^*a^*b^*$** color space.
   - Input: Grayscale Luminance channel ($L$, normalized to $[0, 100]$).
   - Target Output: Chrominance channels ($a^*$ green-red opponent, $b^*$ blue-yellow opponent, normalized to $[-128, 127]$).
   - Recombines $L$ and predicted $a^*, b^*$ and converts back to RGB via `skimage.color.lab2rgb`.
2. **Preprocessing Comparison (Channel Stacking vs. Disk Reloading):**
   - **Method 1 (Direct Channel Stacking):** High contrast and saturated hues, but prone to localized color bleeding/aberrations along high-frequency sky horizons.
   - **Method 2 (Image File Reloading):** Smoother, more natural chromatic transitions with reduced boundary bleeding.
3. **Adversarial & Perturbation Experiments:**
   - **45° Diagonal Rotation:** Induced corner clipping (black bounding triangles) and bilinear interpolation blurring, degrading color prediction fidelity.
   - **Noise Injection:** Flat, low-frequency regions (skies, water, fog) degraded heavily into mottled, pixelated chromatic artifacts, shifting warmer hues into artificial cyan/green casts.
   - **Brightness Perturbations ($\times 0.7$ vs. $\times 1.3$):** Because chrominance predictions are conditioned on input luminance statistics, shifting $L$ outside the training distribution distorted the underlying *hues* (unnatural red/purple halos under darkening; washed-out bluish/cyan tints under overexposure).

---

## 📊 Summary of Robustness & Stress Tests

| Perturbation Type | Cat Haar Cascade | Face Detector (SSD) | Age/Gender (Caffe) | Colorization Autoencoder |
| :--- | :---: | :---: | :---: | :---: |
| **90° / 45° Rotation** | ❌ Complete Failure | ⚠️ Fails except high-contrast | ❌ Dependent on detection | ⚠️ Heavy color & corner distortion |
| **Darkened ($\times 0.5 - 0.7$)** | ⚠️ Minor impact | ⚠️ Fails on shadowed faces | ⚠️ Biases classification | ❌ Generates unnatural red/purple halos |
| **Brightened ($\times 1.3 - 1.8$)** | ⚠️ Minor impact | ✅ Robust localization | ⚠️ Washout biases younger | ❌ Causes cyan/blue desaturation |
| **Aggressive Blur ($51\times51$)** | ❌ Lost edge features | ⚠️ Bounding box degrades | ❌ Catastrophic error (Old → Child) | N/A |
| **Noise Addition** | ⚠️ High false positives | ⚠️ Misses weak edges | ⚠️ High error rate | ❌ Severe chromatic blotching |

---

## 🚀 How to Run

### 1. Prerequisites
Install the required environment packages:

```bash
pip install opencv-python numpy matplotlib tensorflow scikit-image
```

### 2. Running the Notebook
Open Jupyter Lab or Notebook within the `HW3` directory:

```bash
cd HW3
jupyter notebook main.ipynb
```

Execute cells sequentially to:
1. Run cat face detection with Haar Cascades and evaluate rotation/color invariance.
2. Run face, age, and gender DNN inference across static images and generate `Jobs_2_output.mp4`.
3. Load the autoencoder and evaluate CIE $L^*a^*b^*$ colorization and stress tests.
