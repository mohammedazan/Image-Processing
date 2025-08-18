# Tree Species Classification — TRAITEMENT DES DONNEES MULTIMEDIA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9A%A1-brightgreen)](https://pytorch.org/)
[![Open3D](https://img.shields.io/badge/Open3D-Visualization-orange)](http://www.open3d.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn--Machine%20Learning-%2377B900)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-%23FF9900)](https://xgboost.ai/)

---

> **Project Name:** Tree Species Classification from Point Clouds  
> **Short:** A comparative study and implementation of multiple classification pipelines that transform 3D point-cloud scans of individual trees into robust species-level predictions.

---

- [Project Overview](#project-overview)
- [Key Technologies & Logos](#key-technologies--logos)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Methodology & Pipelines](#methodology--pipelines)
  - [Indirect: Multi-view 2D Approaches](#indirect-multi-view-2d-approaches)
  - [Quasi-Direct: 3D Descriptors + Classical ML](#quasi-direct-3d-descriptors--classical-ml)
  - [Direct: Point Cloud Deep Models](#direct-point-cloud-deep-models)
  - [Bonus & Ensembling](#bonus--ensembling)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure (files & folders)](#project-structure-files--folders)
- [Quick Start & Example Commands](#quick-start--example-commands)
- [Design Diagrams & Flowcharts](#design-diagrams--flowcharts)
- [Reproducibility & Experiments Logging](#reproducibility--experiments-logging)
- [Best Practices & Tips](#best-practices--tips)
- [License & Acknowledgements](#license--acknowledgements)

---

# Project Overview

This repository implements a **multi-path classification framework** that converts raw 3D point clouds of trees into species label predictions. The goal is to evaluate and compare a set of *diverse* methods — from lightweight classical pipelines to state-of-the-art point-cloud deep networks — under a consistent preprocessing and evaluation protocol.

Key properties:

- Multi-modal processing: 2D rendered views, handcrafted 3D descriptors, and direct point-cloud learning.
- Reproducible experiments with clear logging and artifact outputs.
- Designed for constrained hardware (e.g. a single NVIDIA GeForce MX110) but compatible with more powerful GPUs.

# Key Technologies & Logos

Below are the main languages, libraries and tools used in the project. *Badges represent brands and give quick access to docs.*

| Category        | Name                                | Purpose                                                  |
| --------------- | ----------------------------------- | -------------------------------------------------------- |
| Language        | **Python 3.8+**                     | Main implementation language, scripting, experiments     |
| Deep Learning   | **PyTorch**                         | Model definition, training, checkpoints                  |
| 3D Processing   | **Open3D**                          | Point cloud IO, visualization, FPS sampling, normals     |
| Rendering       | **pyrender / offscreen**            | Multi-view rendering of point clouds to 2D images        |
| Computer Vision | **torchvision**                     | Pretrained CNN backbones (ResNet / EfficientNet)         |
| ML Toolbox      | **scikit-learn**                    | Baselines, metrics, classical models (SVM, RandomForest) |
| Boosting        | **XGBoost**                         | Fast gradient boosting for tabular features              |
| Logging         | **TensorBoard / WandB (optional)**  | Training curves, hyper-parameter tracking                |
| Utilities       | **OpenCV**                          | Image preprocessing & augmentation                       |

> **Note on trademarks:** Logos and badges above link to official documentation pages. Use of brand images is for illustrative and navigational purposes in the README.

---

# Dataset & Preprocessing

**Dataset**: The dataset contains per-tree 3D point clouds (one folder per species). Class counts vary (e.g., Beech, Red Oak, Ash...). A `test.csv` is provided and remains the fixed external test split.

## Preprocessing steps (highly detailed)

1. **Raw ingestion**
   - Parse directory tree and CSV mappings. Verify counts per species.
   - Expected file formats: `.ply`, `.pcd` or simple `.txt`/`.csv` with XYZ (and optional intensity). Use Open3D to read.

2. **Cleaning & normalization**
   - Remove outliers with statistical filters (e.g., radius/outlier removal).
   - Translate each point cloud to centroid = 0 (centering).
   - Scale to unit sphere or standardized bounding box (scale invariance).

3. **Sampling**
   - Apply **FPS (Farthest Point Sampling)** to sample exactly `N=1024` points per tree for direct point-cloud models. (FPS — French: *échantillonnage par le point le plus éloigné*).

4. **Normals & local features** (if needed)
   - Estimate normals using nearest neighbors (k=16 or 32).
   - Compute local geometric statistics (curvatures, eigenvalues).

5. **Multi-view rendering**
   - Offscreen render `V=12` directional views per tree at `224×224` or `256×256` using either Open3D's offscreen renderer or pyrender.
   - Save rendered views as `.png` with uniform background and optional alpha channel.

6. **Descriptor extraction**
   - 3D descriptors: compute **FPFH**, **VFH** and global histograms (point density, curvature bins).
   - 2D descriptors: compute **HOG** and ORB/SIFT descriptors per rendered view (if allowed).

7. **Caching**
   - All processed artifacts (sampled point clouds, views, descriptors) are saved under `data/processed/` to avoid recomputation.

---

# Methodology & Pipelines

We structure experiments as separate pipelines so each method is reproducible and independently benchmarked.

## Indirect: Multi-view 2D Approaches

**Idea:** Render the 3D point cloud into multiple 2D views and feed them to a 2D CNN backbone.

Options implemented:

1. **End-to-end fine-tuning**
   - Backbone: ResNet50 / EfficientNet-B0 pretrained on ImageNet.
   - Strategy: Freeze backbone initially, train classification head, then unfreeze last `k` layers and fine-tune.

2. **Feature aggregation + classical classifier**
   - Extract feature vector (global average pool) for each view → aggregate views (mean / max / attention-weighted) → single descriptor per object → classify with XGBoost or SVM.

**Strengths & weaknesses**:
- + Leverages strong pretrained image features.
- − Requires rendering overhead and ignores true 3D neighborhood relationships.

## Quasi-Direct: 3D Descriptors + Classical ML

**Idea:** Compute handcrafted 3D descriptors (FPFH, VFH, global histograms) and feed them to light-weight classifiers.

Pipeline:
- Descriptor computation (per-tree) → normalization & PCA (optional) → RandomForest / XGBoost / SVM training.

**Good for baselines and low-GPU regimes.**

## Direct: Point Cloud Deep Models

**Idea:** Feed sampled point clouds straight to networks built for unordered sets.

Supported models:
- **PointNet** (baseline direct model)
- **DGCNN** (graph-based edges — optional if resources permit)

Training specifics:
- Input: `N=1024` points, normalized.
- Augmentations: small rotations (Z or full), jitter, scale, point dropout.
- Loss: Cross-entropy with class weights (to handle imbalance).

## Bonus & Ensembling

- **Stacking:** Use predictions of multiple pipelines as meta-features for an ensemble classifier.
- **Model averaging:** Soft-vote across top-performing models for robustness.

---

# Evaluation Metrics

We evaluate under a single consistent protocol using the fixed `test.csv` split.

| Metric                         | Why it matters                                         |
| ------------------------------ | ------------------------------------------------------ |
| Accuracy                       | Intuitive overall success rate                         |
| Balanced Accuracy              | Adjusts for class imbalance (average recall per class) |
| Macro F1                       | Harmonic mean across classes (equal weight)            |
| Precision / Recall (per-class) | Understand which species are confused                  |
| Confusion Matrix               | Diagnostic tool for systematic errors                  |
| Inference time & Model size    | Practical considerations for deployment                |

**Evaluation notes:** Use the validation split to choose hyperparameters. Only measure final performance once on `test.csv` to avoid leakage.

---

# Project Structure (files & folders)

```text
project_root/
├─ data/
│  ├─ raw/                 # original point clouds and csvs
│  └─ processed/           # sampled point clouds, views, descriptors
├─ notebooks/              # EDA and quick experiments
├─ src/
│  ├─ preprocessing/       # point cloud cleaning, FPS, render_views.py
│  ├─ features/            # descriptor extraction (3D and 2D)
│  ├─ models/              # implementations: pointnet, multiview, classifiers
│  ├─ train.py             # unified training entrypoint
│  └─ eval.py              # produces metrics & confusion matrices
├─ experiments/            # experiment configs + saved checkpoints
├─ results/                # metrics, plots, confusion matrices
├─ requirements.txt
└─ README.md
