# Face Rehabilitation Pipeline — Technical Report

## Overview

This document summarises the complete facial rehabilitation exercise recognition
pipeline built on MediaPipe FaceMesh + CTR-GCN.

---

## Architecture

```
Raw Videos (dataset_raw_face/)
        │
        ▼
MediaPipe FaceMesh (mp.solutions.face_mesh)
  → 468 raw landmarks → filter to 33 physiotherapy nodes
        │
        ▼
Frame-Level CSV  (processed_dataset_face/face_frame_labels.csv)
  columns: video_name, frame, label, landmark_0_x .. landmark_32_z
        │
        ▼
CTR-GCN Tensor Builder
  (T_raw, 33, 3) → resample → (3, 300, 33, 1)
        │
        ▼
Stratified Train/Test Split (80/20)
        │
        ▼
Custom Face Graph  (graph/face_graph.py)
  A.shape = (3, 33, 33)
  K=3 partitions: self / centripetal / centrifugal
        │
        ▼
CTR-GCN Model  (model/ctrgcn.py)
  in_channels=3 | num_point=33 | num_person=1 | num_class=6
        │
        ▼
Training (AdamW + CosineAnnealingLR, 100 epochs)
        │
        ▼
Checkpoint: models/best_face_ctrgcn.pth
Results:    results_face/
```

---

## Landmark Subset (33 Nodes)

Selected landmarks cover the key facial muscle groups relevant to rehabilitation:

| Region | Nodes | Clinical relevance |
|---|---|---|
| Left Eyebrow | 0–4 | Frontalis / corrugator muscle group |
| Right Eyebrow | 5–9 | Frontalis / corrugator muscle group |
| Eyes | 10–17 | Orbicularis oculi (eyelid) |
| Cheeks | 18–21 | Zygomaticus major / nasolabial fold |
| Nose | 22–24 | Levator labii (nasal base movement) |
| Mouth | 25–32 | Orbicularis oris / mentalis / risorius |

---

## Face Graph Topology

Root node: **Node 23** (Nose_tip) — anatomical midface anchor.

Structural groups:
- Eyebrow chains (left: 0→4, right: 5→9)
- Eye corner pairs (10↔11, 12↔13) and eyelid pairs (14↔15, 16↔17)
- Brow–eyelid links (2→14, 7→16)
- Cheek–nose bridge (18→22, 19→22)
- Nasolabial–lip corner (20→25, 21→26)
- Nose chain (22→23→24)
- Mouth contour (25→27→26, 25→28→26)
- Inner lip margin (29↔30)
- Jaw / forehead midline (31→28, 32→22)

---

## Dataset

| Property | Value |
|---|---|
| Classes | 6 |
| Videos | 15 |
| Tensor shape | (3, 300, 33, 1) |
| Feature channels | x, y, z (no visibility) |
| MediaPipe API | mp.solutions.face_mesh |
| Split | 80% train / 20% test (stratified) |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Scheduler | CosineAnnealingLR (T_max=100, eta_min=1e-6) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Epochs | 100 |
| Batch size | 8 |
| Seed | 42 |

---

## Data Augmentation

Applied during training only:
1. Gaussian noise σ=0.01 on all x,y,z channels
2. Temporal flip (p=0.5)
3. Bilateral face mirror — swap left/right anatomically symmetric nodes (p=0.5)

---

## Pre-Flight Validations

Five checks run automatically before training:
1. All tensors are `(3, 300, 33, 1)`
2. Graph is `(3, 33, 33)` with diagonal self-link partition
3. All 6 classes present in training set
4. Model forward pass: `(2,3,300,33,1)` → `(2,6)`
5. No train/test sample overlap

---

## File Index

| File | Purpose |
|---|---|
| `graph/face_landmark_mapping.py` | MP-ID → node-index mapping |
| `graph/face_graph.py` | 33-node adjacency matrix (3,33,33) |
| `preprocessing/extract_face_dataset.py` | FaceMesh extraction → CSV |
| `preprocessing/build_face_ctrgcn_dataset.py` | CSV → (3,300,33,1) tensors |
| `preprocessing/split_face_dataset.py` | Stratified train/test split |
| `dataset/face_loader.py` | PyTorch Dataset + DataLoader |
| `training/train_face_ctrgcn.py` | Full training pipeline |
| `inference/predict_face_video.py` | End-to-end inference |
| `models/best_face_ctrgcn.pth` | Best model checkpoint |
| `results_face/` | Plots and classification report |
