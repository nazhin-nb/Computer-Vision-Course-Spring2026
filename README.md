# Computer Vision Coursework — Spring 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-yellow.svg)](https://gradio.app/)

> **Author:** Nazhin Nikkhahbahrami  
> **Student ID:** 810102530  
> **Institution:** University of Tehran — Spring 2026  
> **Course:** Computer Vision

---

## 📌 Repository Overview

This repository contains the complete portfolio of assignments and the capstone final project for the **Computer Vision (Spring 2026)** course. The coursework spans foundational classical vision algorithms, morphological image processing, statistical and template-based recognition, deep convolutional neural networks (CNNs), demographic inference, autoencoder colorization, and deep photographic inpainting using U-Net architectures with interactive web interfaces.

Each subfolder includes fully executed Jupyter Notebooks, analytical write-ups, datasets/media, and a **dedicated, in-depth README** explaining theoretical concepts, implementation details, and benchmark evaluations.

---

## 📚 Curriculum & Assignments Summary

| Assignment | Topic & Focus Areas | Tech Stack | Status | Detailed Docs |
| :--- | :--- | :--- | :---: | :---: |
| [**HW1**](HW1/) | **Image & Video Processing Fundamentals**<br>Color models (RGB, HSV, Grayscale, Otsu binary), contrast adjustments, histogram equalization vs. stretching, custom 2D convolutions vs. OpenCV, impulse/Gaussian noise, edge detection (Sobel, Laplacian, Canny), and frame-by-frame video restoration. | OpenCV, NumPy, Matplotlib | ✅ Completed | [HW1 README](HW1/README.md) |
| [**HW2**](HW2/) | **End-to-End OCR Pipeline & Template Matching**<br>Synthetic CAPTCHA generation with overlap prevention, multi-stage noise suppression, morphological closing, contour extraction, bounding box merging, left-to-right sorting, and normalized cross-correlation (`TM_CCOEFF_NORMED`) achieving **~91.2% accuracy**. | OpenCV, NumPy, Matplotlib | ✅ Completed | [HW2 README](HW2/README.md) |
| [**HW3**](HW3/) | **Face Detection, Demographics & Deep Colorization**<br>Haar Cascades for cat face detection; multi-task OpenCV DNN (SSD ResNet-10) for human face, age, and gender inference on images/video; self-supervised Convolutional Autoencoder in CIE $L^*a^*b^*$ space; comprehensive robustness stress tests (rotation, blur, lighting, noise). | OpenCV DNN, Caffe, TensorFlow, Scikit-Image | ✅ Completed | [HW3 README](HW3/README.md) |
| [**HW4**](HW4/) | **Natural Scene Classification with Deep CNNs**<br>Multi-class classification (6 landscape/urban classes), data augmentation (flip, rotation, zoom), 4-block Conv2D hierarchy, diagnostic learning curves, and regularization (Dropout + tuned LR + filter scaling) boosting accuracy to **84.5%**. | TensorFlow, Keras, Scikit-Learn | ✅ Completed | [HW4 README](HW4/README.md) |
| [**Final Project**](Final%20Project/) | **Deep Photographic Restoration & Inpainting**<br>Restoring physically degraded historical photos (scratches, dust, tears) using a PyTorch **U-Net with skip connections** on the `openphoto-restore-dataset`. Exceeds target thresholds (**PSNR: 22.73 dB**, **SSIM: 0.8476**) and features an interactive **Gradio web UI**. | PyTorch, Torchvision, Gradio, Scikit-Image | ✅ Completed | [Final Project README](Final%20Project/README.md) |

---

## 📁 Directory Structure

```text
Computer-Vision-Course-Spring2026/
├── HW1/                       # Image fundamentals, spatial filtering, edge detection & video
│   ├── CV-HW1-810102530.ipynb
│   ├── CV_HW1.pdf
│   ├── Pic.jpg, *.mp4
│   └── README.md
├── HW2/                       # Synthetic CAPTCHA OCR pipeline & template matching
│   ├── CA2-final.ipynb
│   ├── CV_HW2.pdf
│   ├── Mapset/                # Alphanumeric reference templates (0-9, a-z)
│   └── README.md
├── HW3/                       # Haar cascades, demographic DNN, & autoencoder colorization
│   ├── main.ipynb
│   ├── CV_HW3.pdf
│   ├── Jobs_2_output.mp4
│   └── README.md
├── HW4/                       # Natural scene classification using deep CNNs & regularization
│   ├── CV-HW4-810102530.ipynb
│   ├── CV-HW4.pdf
│   └── README.md
├── Final Project/             # Deep image restoration via U-Net & interactive Gradio UI
│   ├── CV-FinalProject-810102530.ipynb
│   ├── CV-Final Project.pdf
│   └── README.md
├── LICENSE                    # MIT License
└── README.md                  # Main course repository documentation (this file)
```

---

## 🛠️ Technology Stack & Core Tools

- **Core Languages & Environments:** Python 3.8+, Jupyter Notebook, Google Colab / Kaggle.
- **Deep Learning Frameworks:** PyTorch, Torchvision, TensorFlow, Keras.
- **Computer Vision & Imaging:** OpenCV (`cv2`), Scikit-Image (`skimage`), PIL / Pillow.
- **Data & Scientific Computing:** NumPy, Pandas, Scikit-Learn.
- **Visualization & Deployment:** Matplotlib, Seaborn, Gradio (interactive web demo).

---

## 🚀 Quick Start & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nazhin-nb/Computer-Vision-Course-Spring2026.git
cd Computer-Vision-Course-Spring2026
```

### 2. Install General Dependencies
You can install the core packages across all assignments:

```bash
pip install torch torchvision tensorflow opencv-python numpy matplotlib scikit-image scikit-learn gradio
```

### 3. Running an Assignment
Navigate to any target assignment directory and launch Jupyter Notebook:

```bash
# Example: Running the Final Project
cd "Final Project"
jupyter notebook CV-FinalProject-810102530.ipynb
```

For specific execution details, datasets, or hyperparameters, please consult the individual `README.md` file inside each respective folder.

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).