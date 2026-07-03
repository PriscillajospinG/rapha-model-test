# Face Inference Guide

This document explains how to run inference on a new facial rehabilitation
exercise video using a trained CTR-GCN checkpoint.

---

## Prerequisites

Train the model first:

```bash
python training/train_face_ctrgcn.py
# Checkpoint saved to: models/best_face_ctrgcn.pth
```

---

## Basic Usage

```bash
python inference/predict_face_video.py --video path/to/exercise.mp4
```

## Custom Checkpoint

```bash
python inference/predict_face_video.py \
    --video path/to/exercise.mp4 \
    --model models/best_face_ctrgcn.pth
```

---

## Inference Pipeline

The script runs 4 sequential steps:

```
[1/4]  MediaPipe FaceMesh
       ↓ Extract 33 landmarks (x, y, z) per frame
[2/4]  Build tensor
       ↓ Resample to 300 frames → (3, 300, 33, 1)
[3/4]  Load CTR-GCN checkpoint
       ↓ best_face_ctrgcn.pth
[4/4]  Forward pass
       ↓ Ranked class probabilities
```

---

## Example Output

```
────────────────────────────────────────────────────────────
  CTR-GCN Face Rehabilitation Inference
────────────────────────────────────────────────────────────
  Device     : mps
  Model ckpt : models/best_face_ctrgcn.pth
  Video      : exercise_video.mp4

[1/4] Running MediaPipe FaceMesh …
  Video : exercise_video.mp4
  Frames: 300  |  FPS: 30.0
  Frames extracted : 300
  Failed detections: 2 (0.7%)
      Done in 5.3s

[2/4] Building input tensor …
      Tensor shape : (3, 300, 33, 1)
      (C=3, T=300, V=33, M=1)

[3/4] Loading CTR-GCN checkpoint …
  Checkpoint epoch : 87
  Val acc at save  : 83.33%
  Classes          : {0: 'Eyebrows', 1: 'Eyes', ...}

[4/4] Running inference …

────────────────────────────────────────────────────────────
  RESULT
────────────────────────────────────────────────────────────
  Predicted exercise : EYEBROWS
  Confidence         : 91.24%
  Inference time     : 12.4 ms

  All class probabilities:
   1. Eyebrows                  91.24%  ██████████████████ ◄
   2. Eyes                       4.11%
   3. Frown                      2.38%
   4. Lips                       1.29%
   5. Nose                       0.72%
   6. Side Chicks                0.26%
────────────────────────────────────────────────────────────
```

---

## Supported Video Formats

`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`
