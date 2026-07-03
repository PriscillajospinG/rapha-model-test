# Model Training Guide

This document describes the training pipeline for the CTR-GCN model on the lower-limb physiotherapy exercise skeleton dataset.

---

## 1. Running Training

To train the model, execute the following command:

```bash
python training/train_lower_limb_ctrgcn.py
```

---

## 2. Input Tensor Specification

The model accepts batched skeleton representations with shape:

$$\text{Shape: } (N, C, T, V, M)$$

Where:
- $N$ = Batch size (default: `8`)
- $C = 4$ Channels:
  - Channel `0`: $x$ (normalized coordinate)
  - Channel `1`: $y$ (normalized coordinate)
  - Channel `2`: $z$ (relative depth coordinate)
  - Channel `3`: $visibility$ (detection score)
- $T = 300$ Frames
- $V = 10$ Joints re-mapped from MediaPipe Pose landmarks
- $M = 1$ Person

---

## 3. Joint Mapping & Connectivity Graph

### 10 Lower-Limb Joint Indices
The 10 joints are mapped as follows:
- **0**: Left Hip (MediaPipe Index 23)
- **1**: Right Hip (MediaPipe Index 24)
- **2**: Left Knee (MediaPipe Index 25)
- **3**: Right Knee (MediaPipe Index 26)
- **4**: Left Ankle (MediaPipe Index 27)
- **5**: Right Ankle (MediaPipe Index 28)
- **6**: Left Heel (MediaPipe Index 29)
- **7**: Right Heel (MediaPipe Index 30)
- **8**: Left Foot Index (MediaPipe Index 31)
- **9**: Right Foot Index (MediaPipe Index 32)

### Custom Graph Connections
Spatial graph convolutions require a defined adjacency matrix representing physical body connections:
- **Left Leg Branch**: Hip (0) → Knee (2) → Ankle (4) → Heel (6) → Foot Index (8)
- **Right Leg Branch**: Hip (1) → Knee (3) → Ankle (5) → Heel (7) → Foot Index (9)
- **Bilateral Trunk Connection**: Hip (0) ↔ Hip (1)

This graph structure maps to a spatial partition adjacency tensor `A` of shape `(3, 10, 10)` dividing node relationships into:
1. Self-connections
2. Centripetal connections (closer to body trunk)
3. Centrifugal connections (further from body trunk)

---

## 4. Pre-Flight Validations

Before running training gradients, the script performs automated safety checks:
1. **Tensor Shape Verification**: Validates that all `.npy` files referenced in the CSV splits are exactly `(4, 300, 10, 1)`.
2. **Graph Consistency Check**: Verifies that the custom adjacency matrix has the shape `(3, 10, 10)` and correct diagonal self-links.
3. **Class Distribution Check**: Confirms all 9 classes exist in the training dataset and lists their sample counts.
4. **Dry-Run Forward Pass**: Feeds a synthetic dummy batch `(2, 4, 300, 10, 1)` into CTR-GCN to verify the output shape is `(2, 9)` and parameters are fully initialized.

---

## 5. Hyperparameter Settings

The pipeline uses the following settings optimized for fast training on CPU/MPS/CUDA:
- **Optimizer**: `AdamW` (learning rate = $1 \times 10^{-3}$, weight decay = $1 \times 10^{-4}$)
- **Scheduler**: `CosineAnnealingLR` (T_max = 100 epochs, eta_min = $1 \times 10^{-6}$)
- **Loss Function**: Cross-entropy with $0.1$ label smoothing
- **Batch Size**: `8`
- **Epochs**: `100`
- **Data Augmentations**: Real-time Gaussian noise ($\sigma=0.01$), temporal reversing, and bilateral mirroring (left-right joint swaps) applied on training batches.

---

## 6. Training Outputs

Once execution finishes, the pipeline produces the following outputs:
- **Model Checkpoints**:
  - `models/best_lower_limb_ctrgcn.pth`: PyTorch checkpoint containing the state dict of the model, optimizer, best epoch, and validation accuracy.
- **Evaluation Visualizations**:
  - `results/loss_curve.png`: Plot of training loss over 100 epochs.
  - `results/accuracy_curve.png`: Comparison plot of training vs. validation accuracy.
  - `results/confusion_matrix.png`: Annotated heat-map matrix of class-wise predictions on the test split.
- **Log Files**:
  - `training.log`: Comprehensive text log containing pre-flight validations, per-epoch performance statistics, and the final classification report.
