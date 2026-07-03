# Upper-Limb CTR-GCN Training

## Overview

The upper-limb CTR-GCN model is trained to classify physiotherapy exercises from 8-joint upper-limb skeleton sequences.

---

## Architecture

### Input Tensor

| Dimension | Value | Meaning |
|---|---|---|
| N | batch size | samples per batch |
| C | 4 | channels: x, y, z, visibility |
| T | 300 | temporal frames |
| V | 8 | graph nodes (upper-limb joints) |
| M | 1 | persons |

### Custom Upper-Limb Graph

File: `graph/upper_limb.py`

**Nodes (8):**

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

**Connections:**
- Shoulder bar: `0 ↔ 1`
- Left arm: `0–2 · 2–4`
- Right arm: `1–3 · 3–5`
- Trunk anchors: `0–6 · 1–7`
- Hip bar: `6–7`

**Adjacency matrix:** `A.shape = (3, 8, 8)` — 3 ST-GCN partitions (self / centripetal / centrifugal)

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

1. **Tensor shapes** — Every `.npy` in train + test is exactly `(4, 300, 8, 1)`
2. **Graph properties** — `A.shape == (3, 8, 8)`, `A[0]` is diagonal
3. **Class coverage** — Warns if any class has no training samples
4. **Forward pass** — Synthetic `(2, 4, 300, 8, 1)` batch → `(2, num_classes)` output

Training only starts after **all 4 checks pass**.

---

## Running Training

```bash
cd labeling-cv
source .venv/bin/activate
python training/train_upper_limb_ctrgcn.py
```

---

## Outputs

| File | Description |
|---|---|
| `models/best_upper_limb_ctrgcn.pth` | Best validation accuracy checkpoint |
| `results_upper/loss_curve.png` | Training loss over epochs |
| `results_upper/accuracy_curve.png` | Train vs. validation accuracy |
| `results_upper/confusion_matrix.png` | Per-class confusion matrix (test set) |
| `upper_training.log` | Full training log |

---

## Known Limitations

With a small dataset (~53 videos, 4–5 classes), the model is likely to overfit. The 2.16M parameter CTR-GCN model generalises better with more training data. Expanding the dataset is strongly recommended for clinical deployment.
