# Face Training Guide

This document explains how to train the CTR-GCN model for facial rehabilitation
exercise recognition.

---

## Prerequisites

Complete the dataset creation steps first:

```bash
python preprocessing/extract_face_dataset.py
python preprocessing/build_face_ctrgcn_dataset.py
python preprocessing/split_face_dataset.py
```

Install training dependencies:

```bash
pip install torch matplotlib seaborn scikit-learn
```

---

## Running Training

```bash
python training/train_face_ctrgcn.py
```

---

## Pre-Flight Validation (5 checks)

Before any gradient step, the script automatically validates:

| Check | What is tested |
|---|---|
| ① Tensor shapes | Every `.npy` in train+test is `(3, 300, 33, 1)` |
| ② Graph adjacency | `A.shape == (3, 33, 33)`, A[0] is diagonal |
| ③ Class coverage | All 6 classes are represented in the training set |
| ④ Model forward pass | Synthetic batch `(2, 3, 300, 33, 1)` → output `(2, 6)` |
| ⑤ Train/test overlap | No sample appears in both splits |

If any check fails, training aborts with a descriptive error message.

---

## Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Epochs | 100 |
| Batch size | 8 |
| Augmentation | Gaussian noise + temporal flip + bilateral face mirror |

---

## Model Architecture

The model reuses the full CTR-GCN architecture (`model/ctrgcn.py`) unchanged
with face-specific configuration:

| Parameter | Value |
|---|---|
| `num_point` | 33 |
| `num_person` | 1 |
| `in_channels` | 3 |
| `num_class` | 6 (dynamic) |
| Graph | `graph/face_graph.py` |

---

## Outputs

| File | Description |
|---|---|
| `models/best_face_ctrgcn.pth` | Best validation accuracy checkpoint |
| `results_face/loss_curve.png` | Training loss over epochs |
| `results_face/accuracy_curve.png` | Train vs. validation accuracy |
| `results_face/confusion_matrix.png` | Per-class confusion matrix (test set) |
| `results_face/classification_report.txt` | Precision/recall/F1 per class |
| `face_training.log` | Full training log |

---

## Face Graph

The `FaceGraph` class (`graph/face_graph.py`) constructs a 33-node facial
skeleton adjacency tensor `A` of shape `(3, 33, 33)`:

- **A[0]** — Self-links (identity)
- **A[1]** — Centripetal links (toward root = node 23, Nose_tip)
- **A[2]** — Centrifugal links (away from root)

Anatomical connections include eyebrow chains, eye corner pairs, nasolabial
links, mouth contour, and jaw/forehead midline anchors.
