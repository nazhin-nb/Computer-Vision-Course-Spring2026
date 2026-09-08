# Computer Vision — Homework 2: End-to-End OCR Pipeline & Template Matching

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange.svg)](https://numpy.org/)
[![Course](https://img.shields.io/badge/Course-Computer_Vision_Spring_2026-purple.svg)](#)

> **Author:** Nazhin Nikkhahbahrami  
> **Student ID:** 810102530  
> **Course:** Computer Vision (Spring 2026)

---

## 📌 Overview

This repository contains the complete implementation and empirical analysis for **Homework 2** of the Computer Vision course. The project designs and evaluates an end-to-end **Optical Character Recognition (OCR)** system for synthetic noisy word images (CAPTCHAs). The pipeline spans synthetic data generation with non-overlapping characters, multi-stage noise suppression and edge restoration, connected-component filtering, contour-based character segmentation with spatial ordering, and normalized cross-correlation template matching achieving an average accuracy of **~91.2%**.

---

## 📁 Repository Structure

```text
HW2/
├── CA2-final.ipynb          # Comprehensive Jupyter Notebook with code, stages & analysis
├── CV_HW2.pdf               # Homework assignment description
├── Mapset/                  # Standard alphanumeric reference template dataset
│   ├── 0.png ... 9.png      # Digit templates (10 classes)
│   └── a.png ... z.png      # Lowercase letter templates (26 classes)
└── README.md                # Project documentation
```

---

## 🔬 Homework Breakdown

### Part 1: Producing Synthetic Data & Noise Modeling

1. **Synthetic Word Synthesis:**
   - Samples 4 random alphanumeric characters from the `Mapset/` dataset to construct realistic multi-character words.
   - **Overlap Prevention:** Dynamically computes character bounding boxes and advances horizontal placement based on character width plus controlled randomized spacing, ensuring distinct letter separation.
2. **Noise Simulation:**
   - Corrupts generated images with **Salt & Pepper noise** (`amount=0.05`, `salt_vs_pepper=0.5`).
   - Analyzes degradation effects: impulse noise corrupts both the high-contrast background and internal character strokes.
3. **Initial Denoising with Median Filter:**
   - Evaluates median filtering ($3\times3$ vs. $5\times5$ window) to eliminate impulse noise while preserving fundamental character geometry.

---

### Part 2: Image Preprocessing & Morphological Refinement

1. **Filtering Comparison (Median vs. Gaussian Blur):**
   - Investigates why blurring suppresses high spatial frequencies.
   - Demonstrates why Gaussian filtering fails on impulse noise (weighted averaging merely smears high-magnitude outliers into cloudy gray patches), whereas non-linear Median filtering cleanly rejects outlier spikes.
2. **Edge Sharpening vs. De-blurring:**
   - **Sharpening Selection:** Median filtering introduces slight boundary softening. A high-pass sharpening kernel is applied to restore high-contrast, crisp boundaries essential for contour localization.
   - **Avoiding De-blurring Risks:** Inverse de-blurring risks amplifying residual high-frequency noise and inducing ringing artifacts, which impairs subsequent thresholding.
3. **Connected Component Noise Filtering (Area Thresholding):**
   - Identifies remaining isolated noise blobs using connected component analysis.
   - **Hyperparameter Tuning:**
     - A 25-pixel threshold eliminated noise but inadvertently discarded critical small character details (such as the dots of letters `'i'` and `'j'`).
     - Tuning the area threshold to **10 pixels** established the optimal trade-off: safely removing spurious noise specks while preserving valid miniature character features.

---

### Part 3: Character Segmentation & Normalization

1. **Binarization & Morphological Grouping:**
   - Inverts and binarizes preprocessed text images using Otsu thresholding.
   - Applies vertical morphological closing (`cv2.morphologyEx` with vertical kernel $15\times1$) to connect vertically disjointed parts of single characters (e.g., reconnecting `'i'` and `'j'` dots to stems when needed).
2. **Contour Extraction & Bounding Box Merging:**
   - Extracts character contours using `cv2.findContours(cv2.RETR_EXTERNAL)`.
   - Implements a custom box merging mechanism (`merge_bounding_boxes`) to resolve over-segmentation and merge vertically split components of individual letters.
3. **Left-to-Right Ordering & Dimensional Standardization:**
   - Sorts detected bounding boxes strictly by ascending X-coordinates ($x_{\text{min}}$) to ensure correct character reading order.
   - Crops each segmented letter, applies boundary padding, and normalizes dimensions to a uniform **$64 \times 64$** pixel resolution using area interpolation (`cv2.INTER_AREA`).

---

### Part 4: Character Recognition via Template Matching

1. **Template Preparation & Zero-Padding:**
   - Loads the 36 ground-truth templates from `Mapset/`, resizes to $64 \times 64$, and applies an **8-pixel zero-padding border** (`cv2.copyMakeBorder`), expanding reference dimensions to **$80 \times 80$** pixels.
   - This spatial margin accommodates translation offsets and crop misalignments during character matching.
2. **Normalized Cross-Correlation Matching:**
   - Employs OpenCV's `cv2.matchTemplate` using the **`cv2.TM_CCOEFF_NORMED`** metric:
     $$\mathcal{R}(x, y) = \frac{\sum_{x',y'} \left(T'(x',y') \cdot I'(x+x',y+y')\right)}{\sqrt{\sum_{x',y'} T'^2(x',y') \cdot \sum_{x',y'} I'^2(x+x',y+y')}}$$
   - Invariant to global brightness variations; picks the class yielding the highest correlation coefficient.
3. **Automated Evaluation & Logging:**
   - Automatically compares predictions against true labels parsed from generated file names and exports results to CSV.

---

## 📊 Performance & Benchmark Evaluation

### Multi-Run System Accuracy (10 Independent Runs)

| Run # | Generated Words (4-chars each) | Accuracy (%) |
| :---: | :---: | :---: |
| 1 | 20 words (80 chars) | **90.00%** |
| 2 | 20 words (80 chars) | **91.25%** |
| 3 | 20 words (80 chars) | **91.25%** |
| 4 | 20 words (80 chars) | **92.50%** |
| 5 | 20 words (80 chars) | **95.00%** |
| 6 | 20 words (80 chars) | **86.25%** |
| 7 | 20 words (80 chars) | **89.87%** |
| 8 | 20 words (80 chars) | **90.00%** |
| 9 | 20 words (80 chars) | **93.75%** |
| 10 | 20 words (80 chars) | **92.50%** |
| **Average** | **200 words (800 chars)** | **91.24%** |

### 🔍 Failure Modes & Confusion Analysis

Because template matching relies on pixel-wise normalized cross-correlation rather than high-level semantic features, errors typically cluster among topologically similar characters:

- **`'s'` vs. `'8'`:** Shared double-curved silhouette; residual boundary blur from smoothing can bridge small loops, causing `'s'` to match `'8'`.
- **`'0'` (Zero) vs. `'o'` (Letter O):** In standard geometric typefaces, these characters are nearly identical in aspect ratio and stroke shape.
- **`'l'` vs. `'i'` vs. `'j'`:** Vertical bar ambiguity; if the dot above `'i'` or `'j'` is fragmented or filtered out during segmentation, the remaining stem closely correlates with `'l'`.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install opencv-python numpy matplotlib
```

### 2. Running the Notebook
Launch Jupyter Lab or Notebook within the `HW2` directory:

```bash
cd HW2
jupyter notebook CA2-final.ipynb
```

Execute all cells sequentially to:
1. Synthesize random CAPTCHA words from `Mapset/`.
2. Apply the end-to-end denoising, sharpening, and connected-component cleaning pipeline.
3. Segment characters, sort from left to right, and standardize to $64\times64$.
4. Run template matching against $80\times80$ padded references and generate the classification accuracy report.
