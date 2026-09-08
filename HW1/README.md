# Computer Vision — Homework 1: Image & Video Processing Fundamentals

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange.svg)](https://numpy.org/)
[![Course](https://img.shields.io/badge/Course-Computer_Vision_Spring_2026-purple.svg)](#)

> **Author:** Nazhin Nikkhahbahrami  
> **Student ID:** 810102530  
> **Course:** Computer Vision (Spring 2026)

---

## 📌 Overview

This repository contains the implementation and analysis for **Homework 1** of the Computer Vision course. The project covers core digital image processing fundamentals, spatial filtering techniques (both custom from-scratch implementations and optimized OpenCV functions), noise modeling and removal, edge detection, and frame-by-frame video restoration.

---

## 📁 Repository Structure

```text
HW1/
├── CV-HW1-810102530.ipynb   # Complete Jupyter Notebook with code, plots & analysis
├── CV_HW1.pdf               # Homework assignment description
├── Pic.jpg                  # Input image for Parts 1 & 2
├── Original_Vid.mp4         # Clean source video for Part 3
├── Noisy_Vid.mp4            # Video corrupted with Salt & Pepper noise
├── Filtered_Vid.mp4         # Restored video using Median Filtering
└── README.md                # Project documentation
```

---

## 🔬 Homework Breakdown

### Part 1: Image Representations, Color Spaces & Histograms

1. **Color Space Transformations:**
   - **BGR → RGB:** Color channel reordering for correct Matplotlib rendering.
   - **Grayscale:** Single-channel intensity reduction preserving structural details (`cv2.COLOR_BGR2GRAY`).
   - **Binarization:** Adaptive thresholding using **Otsu’s method** (`cv2.THRESH_OTSU`) for foreground/background segmentation.
   - **HSV Color Model:** Decomposing images into Hue (chroma), Saturation (vibrancy), and Value (brightness) for illumination-invariant processing.

2. **Contrast Measurement & Adjustment:**
   - Quantitative evaluation via standard deviation ($\sigma$) of intensity.
   - **Linear Scaling:** $I_{\text{out}} = \alpha \cdot I_{\text{in}} + \beta$ (contrast and brightness adjustment).
   - **Non-linear Gamma Correction:** Power-law transformation $I_{\text{out}} = 255 \cdot (I_{\text{in}} / 255)^\gamma$.

3. **Histogram Analysis & Enhancement:**
   - Visualizing 1D intensity distributions for Grayscale and separate R, G, B channels.
   - **Histogram Equalization:** Spreading out frequent intensities to achieve a flat CDF (`cv2.equalizeHist`).
   - **Histogram Stretching (Min-Max Normalization):** Linearly stretching contrast across the full dynamic range $[0, 255]$ (`cv2.normalize`).

---

### Part 2: Spatial Filtering, Noise Modeling & Edge Detection

1. **Noise Simulation:**
   - **Salt & Pepper Noise:** Random impulse noise at $5\%$, $10\%$, and $20\%$ corruption densities.
   - **Gaussian Noise:** Additive normal distribution noise with varying standard deviation ($\sigma$).

2. **Custom vs. Built-in Spatial Convolutions:**
   - Implemented custom sliding-window convolution (`custom_convolve`) and non-linear median filtering (`custom_median_filter`) from scratch using NumPy.
   - Compared visual output and execution efficiency against OpenCV primitives (`cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur`).

3. **Impulse Noise Restoration (Why Median Filter Wins):**
   - **Mean / Gaussian Filters (Linear):** Average extreme outliers ($0$ and $255$) with neighbors, smearing noise into blurry artifacts.
   - **Median Filter (Non-linear Order Statistic):** Selects the statistical median within the window, completely rejecting extreme impulse outliers while preserving sharp boundaries and structural edges.

4. **Edge Detection Operators:**
   - **Sobel Operator:** First-order gradient kernels ($G_x, G_y$) calculating magnitude $\sqrt{G_x^2 + G_y^2}$ and edge orientation.
   - **Laplacian Operator:** Second-order isotropic derivative detecting zero-crossings (sensitive to noise).
   - **Canny Edge Detector:** Multi-stage optimal detector incorporating Gaussian smoothing, gradient computation, non-maximum suppression (NMS), and hysteresis thresholding.

5. **Comprehensive Filter Benchmark on Noisy Images:**
   - Tested Mean ($3\times3, 5\times5$), Gaussian ($3\times3, 5\times5$), Median ($3\times3, 5\times5$), Bilateral, Min/Max, Laplacian, Sobel, and Unsharp Masking filters.

---

### Part 3: Video Processing & Spatiotemporal Denoising

1. **Video Stream Handling:**
   - Decoded `Original_Vid.mp4` properties (FPS, resolution, total frame count) via `cv2.VideoCapture`.
2. **Sequential Noise Injection:**
   - Generated `Noisy_Vid.mp4` by introducing synthetic Salt & Pepper noise per frame while preserving video timing and codec parameters via `cv2.VideoWriter`.
3. **Sequential Median Denoising:**
   - Restored corrupted frames using median filtering to produce `Filtered_Vid.mp4`.
   - Analyzed trade-offs: significant impulse noise reduction vs. slight softening of fine texture details.

---

## 📊 Summary of Key Findings

| Operation / Filter | Primary Purpose | Edge Preservation | Salt & Pepper Removal | Speed |
| :--- | :--- | :---: | :---: | :---: |
| **Mean Filter** | General smoothing / Blur | Poor | Poor (blurs noise) | Fast |
| **Gaussian Filter** | Natural smoothing | Moderate | Poor (spreads noise) | Fast |
| **Median Filter** | Impulse noise reduction | **High** | **Excellent** | Moderate |
| **Bilateral Filter** | Edge-preserving smoothing | High | Fair | Slower |
| **Sobel / Canny** | Edge detection / Boundaries | N/A (Highlights) | N/A (Amplifies noise) | Fast |

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install opencv-python numpy matplotlib
```

### 2. Running the Notebook
Launch Jupyter Lab or Notebook within the `HW1` directory:

```bash
cd HW1
jupyter notebook CV-HW1-810102530.ipynb
```

Run all cells sequentially to reproduce the visualizations, histograms, filter comparisons, and output videos (`Noisy_Vid.mp4` and `Filtered_Vid.mp4`).
