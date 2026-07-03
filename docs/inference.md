# Inference & Video Prediction Guide

This document describes how to execute predictions on new, unseen videos using the trained CTR-GCN model.

---

## 1. Running Inference

To run predictions on a physiotherapy video, use the CLI script `predict_video.py` located in the `inference` folder:

```bash
python inference/predict_video.py --video path/to/new_video.mp4
```

### Script Arguments:
- `--video` (required): Absolute or relative path to the target video file (`.mp4`, `.avi`, `.mov`, etc.).
- `--model` (optional): Path to a custom CTR-GCN `.pth` checkpoint. Defaults to `models/best_lower_limb_ctrgcn.pth`.

---

## 2. End-to-End Inference Flow

When you execute the command, the script runs the following steps sequentially:

```text
Raw Video File (.mp4)
      │
      ▼  [Step 1: MediaPipe Pose Extraction]
Extract raw frame-by-frame coordinates for joints 23-32 -> Shape: (T_raw, 10, 4)
      │
      ▼  [Step 2: Tensor Construction]
Resample/pad sequences to exactly 300 frames, transpose -> Shape: (4, 300, 10, 1)
      │
      ▼  [Step 3: Load Model & Weights]
Build CTR-GCN graph structures and load best checkpoint weights
      │
      ▼  [Step 4: Forward Pass & Softmax]
Inference forward pass through network, compute class probabilities
      │
      ▼  [Step 5: Visual Reports]
Print top predicted exercise, confidence score, and rank-ordered probabilities
```

---

## 3. Detailed Step Breakdown

### Step 1: MediaPipe Pose Extraction
- Uses the `mediapipe` library to detect the person's skeleton frame-by-frame.
- Extracts coordinates $x, y, z$ and visibility for the 10 lower-limb joints (hips, knees, ankles, heels, and feet).
- Frames without a detected skeleton are initialized to all-zeros.

### Step 2: Temporal Resampling & Normalization
- Downsamples or tiles the temporal sequence to target $T = 300$ frames.
- Re-maps the coordinates to $(C, T, V, M)$ shape, where $C=4, T=300, V=10, M=1$.
- Converts the final data array into a PyTorch FloatTensor.

### Step 3: Forward Pass & Classification
- Loads the neural network architecture configured with a custom 10-node body connectivity graph.
- Executes the forward pass to obtain logits, then applies Softmax to calculate confidence percentages for each of the 9 exercises.

---

## 4. Probability Output Explanation

The terminal output renders a clean visualization of predictions:

```text
───────────────────────────────────────────────────────
  CTR-GCN Lower-Limb Inference
───────────────────────────────────────────────────────
  Device     : mps
  Model ckpt : models/best_lower_limb_ctrgcn.pth
  Video      : path/to/sample.mp4

[1/4] Running MediaPipe Pose …
  Video : sample.mp4
  Frames: 180  |  FPS: 30.0
  Frames extracted : 180
  Failed detections: 0 (0.0%)
      Done in 1.4s

[2/4] Building input tensor …
      Tensor shape : (4, 300, 10, 1)

[3/4] Loading CTR-GCN checkpoint …
  Checkpoint epoch : 85
  Val acc at save  : 53.85%

[4/4] Running inference …

───────────────────────────────────────────────────────
  RESULT
───────────────────────────────────────────────────────
  Predicted exercise : KNEE
  Confidence         : 84.62%
  Inference time     : 12.3 ms

  All class probabilities:
     1. knee          84.62%  ████████████████ ◄
     2. quadriceps    10.15%  ██
     3. leg_raise      3.22%  
     4. hamstring      1.01%  
     5. heel_slide     0.50%  
     6. hip            0.30%  
     7. calf           0.10%  
     8. toes           0.08%  
     9. ankle          0.02%  
───────────────────────────────────────────────────────
```

- **Predicted exercise**: The class with the highest probability value.
- **Confidence**: Softmax probability percentage associated with the predicted class.
- **Inference time**: Net computation time for the PyTorch forward pass (excluding MediaPipe setup).
- **All class probabilities**: Sorted list of classes with visual horizontal bars indicating confidence levels.
