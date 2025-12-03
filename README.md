# Comparative Analysis of Dimensionality Reduction Methods for Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)

**Authors:** Sahil Dadhwal, Purva Khadke, Sarvesh Halbe  
**Course:** ECS 271 - Machine Learning

---

## Table of Contents
* [Project Overview](#-project-overview)
* [Key Features](#-key-features)
* [Experimental Pipeline](#-experimental-pipeline)
* [Installation](#-installation)
* [Usage](#-usage)
* [Results](#-results)
* [Conclusion](#-conclusion)
---

## Project Overview
High-dimensional image data poses significant computational challenges for modern machine learning models. This project investigates the trade-off between **classification accuracy** and **computational efficiency** by applying various dimensionality reduction techniques to the **CIFAR-100** dataset.

We benchmark **Linear (PCA)** vs. **Nonlinear (UMAP, Autoencoder)** reduction methods across three distinct classifier architectures:
1.  **ResNet-50** (Transfer Learning)
2.  **Vision Transformer** (Custom, Trained from Scratch)
3.  **Autoencoder Classifier** (Joint Reconstruction & Classification)

### The Core Problem
Selecting the right reduction technique is critical because the distinction between linear and nonlinear approaches impacts the final classification performance. Our goal is to identify which method best preserves classification accuracy and to determine whether modern Transformer architectures can outperform traditional Autoencoders in this domain.

---

## Key Features
* **Comprehensive Benchmark:** Evaluates 12 unique experimental conditions (4 Inputs × 3 Models).
* **Dataset:** CIFAR-100 (60,000 images, 100 classes).
* **Dimensionality Reduction:** Compresses inputs from **3,072** dimensions down to **512**.
* **Model Variety:** Compares standard CNNs (ResNet) against modern Vision Transformers (ViT) and Deep Autoencoders.
* **Metric Analysis:** Detailed tracking of Top-1 Accuracy, Training Time, and Reconstruction Loss.

---

## Experimental Pipeline

The following table outlines the end-to-end workflow implemented in this project, from raw data ingestion to final model evaluation.

| Stage | Component | Description & Configuration |
| :--- | :--- | :--- |
| **1. Data Input** | **CIFAR-100** | Loading 60,000 images across 100 classes (Split: 50k Train / 10k Test). |
| **2. Preprocessing** | **Flatten & Scale** | Images are flattened from $32\times32\times3$ to **3,072-D vectors** and normalized to $[0, 1]$. |
| **3. Dimensionality Reduction** | **Compression** | Reducing features from **3,072 $\rightarrow$ 512 dimensions** using PCA, UMAP, or Autoencoders. |
| **4. Classification** | **Model Training** | Training ResNet-50 (Transfer Learning), Vision Transformer, or AE Classifier on the reduced data. |
| **5. Evaluation** | **Metrics** | Measuring Classification Accuracy (%), Training Time (seconds), and Reconstruction Loss. |

---

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### Setup
1.  **Clone the repository**
    ```bash
    cd CNN_FeatureReduction
    git clone https://github.com/purvakhadke/CNN_FeatureReduction.git
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

All source code is located in the `code` folder. To run the experiments, navigate to that directory and execute the `main.py` script.

```bash
cd code
```
## Results

| **Stage** | **Component** | **Description & Configuration** |
| :--- | :--- | :--- |
| **1. Classifier** | **ResNet-50 / Transformer / AE Classifier** | Evaluation of three model families across multiple input representations. |
| **2. Input Data** | **Raw / PCA / UMAP** | Models trained on **3,072-D raw vectors** or **512-D compressed features** (PCA or UMAP). |
| **3. Dimensions** | **Feature Size** | Raw images: **3072 dimensions**; Compressed representations: **512 dimensions**. |
| **4. Performance** | **Accuracy (%)** | ResNet-50 (Raw): **65.82%** (highest). Transformer (Raw): **35.45%**. PCA inputs generally improved stability versus UMAP. |
| **5. Efficiency** | **Training Time** | Fastest: ResNet-50 (PCA / UMAP): **~2 min**. Slowest: Transformer (Raw): **~136 min**. |
| **6. Insight** | **Observations** | Raw ResNet-50 performs best; PCA drastically reduces training time; UMAP harms structure; Transformer struggles without pretraining; AE fastest but least accurate among major models. |

## Conclusion

Our experiments confirm that transfer learning with ResNet-50 on raw data yields the highest accuracy (65.82%), significantly outperforming all dimensionality reduction methods. While techniques like PCA reduced training time by nearly 98%, they incurred a substantial accuracy loss, with UMAP failing to preserve class-discriminative features in high dimensions. Ultimately, dimensionality reduction serves as a viable trade-off only when computational speed is strictly prioritized over predictive precision.
