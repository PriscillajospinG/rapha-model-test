# Post-Stroke CTR-GCN Training

## Overview

The post-stroke CTR-GCN model is trained to classify physiotherapy exercises from 12-joint skeleton sequences.

---

## Architecture

### Input Tensor

| Dimension | Value | Meaning |
|---|---|---|
| N | batch size | samples per batch |
| C | 4 | channels: x, y, z, visibility |
| T | 300 | temporal frames |
| V | 12 | graph nodes (post-stroke joints) |
| M | 1 | persons |

### Custom Post-Stroke Graph

File: `graph/poststroke_graph.py`

**Nodes (12):**

| Node | Joint |
|---|---|
| 0 | Left Shoulder |
| 1 | Right Shoulder |
| 2 | Left Elbow |
| 3 | Right Elbow |
| 4 | Left Wrist |
| 5 | Right Wrist |
| 6 | Left Hip |
| 7 | Right Hip |
| 8 | Left Knee |
| 9 | Right Knee |
| 10 | Left Ankle |
| 11 | Right Ankle |

**Connections:**
- Left side: $0 \rightarrow 2 \rightarrow 4$ (shoulder $\rightarrow$ elbow $\rightarrow$ wrist) and $6 \rightarrow 8 \rightarrow 10$ (hip $\rightarrow$ knee $\rightarrow$ ankle)
- Right side: $1 \rightarrow 3 \rightarrow 5$ (shoulder $\rightarrow$ elbow $\rightarrow$ wrist) and $7 \rightarrow 9 \rightarrow 11$ (hip $\rightarrow$ knee $\rightarrow$ ankle)
- Trunk: $0 \leftrightarrow 1$, $6 \leftrightarrow 7$
- Body connections: $0 \leftrightarrow 6$, $1 \leftrightarrow 7$
- Cross stabilization: $6 \leftrightarrow 8$, $7 \leftrightarrow 9$

**Adjacency matrix:** `A.shape = (3, 12, 12)` — 3 ST-GCN partitions (self / centripetal / centrifugal)

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Batch size | 8 |
| Epochs | 100 |
| Augmentation | Gaussian noise + temporal flip + L↔R mirror |

---

## Pre-Flight Validation

Before any gradient step, the training script automatically verifies:

1. **Tensor shapes** — Every `.npy` in train + test is exactly `(4, 300, 12, 1)`
2. **Graph properties** — `A.shape == (3, 12, 12)`, `A[0]` is diagonal
3. **Class coverage** — Warns if any class has no training samples
4. **Train/Test overlap** — Confirms that no sample is shared between training and test sets
5. **Forward pass** — Synthetic `(2, 4, 300, 12, 1)` batch → `(2, num_classes)` output

Training only starts after **all 5 checks pass**.

---

## Running Training

```bash
python training/train_poststroke_ctrgcn.py
```

---

## Outputs

| File | Description |
|---|---|
| `models/best_poststroke_ctrgcn.pth` | Best validation accuracy checkpoint |
| `results_poststroke/loss_curve.png` | Training loss over epochs |
| `results_poststroke/accuracy_curve.png` | Train vs. validation accuracy |
| `results_poststroke/confusion_matrix.png` | Per-class confusion matrix (test set) |
| `results_poststroke/classification_report.txt` | Per-class performance details |
| `poststroke_training.log` | Full training log |
