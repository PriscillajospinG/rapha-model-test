# Upper-Limb CTR-GCN Inference

## Overview

Run the trained model on any new upper-limb physiotherapy video to get an exercise prediction with confidence scores.

---

## Usage

```bash
cd labeling-cv
source .venv/bin/activate

python inference/predict_upper_video.py --video path/to/exercise.mp4
```

### Optional: specify a custom checkpoint

```bash
python inference/predict_upper_video.py \
    --video  path/to/exercise.mp4 \
    --model  models/best_upper_limb_ctrgcn.pth
```

---

## Inference Pipeline

```
Input Video
    ↓
[1/4] MediaPipe Pose — extracts joints 11, 12, 13, 14, 15, 16, 23, 24
    ↓
[2/4] Build tensor (C=4, T=300, V=8, M=1)
      • frames > 300 → uniform temporal sampling
      • frames < 300 → tile/loop to reach 300
    ↓
[3/4] Load best_upper_limb_ctrgcn.pth
    ↓
[4/4] CTR-GCN forward pass → softmax probabilities
    ↓
Display: Predicted class + confidence + ranked probability bar chart
```

---

## Example Output

```
──────────────────────────────────────────────────────────
  CTR-GCN Upper-Limb Inference
──────────────────────────────────────────────────────────
  Device     : mps
  Model ckpt : models/best_upper_limb_ctrgcn.pth
  Video      : Shoulder-Lateral-Rotation.mp4

[1/4] Running MediaPipe Pose on upper-limb joints …
  Video : Shoulder-Lateral-Rotation.mp4
  Frames: 240  |  FPS: 30.0
  Frames extracted : 240
  Failed detections: 3 (1.2%)
      Done in 4.3s

[2/4] Building input tensor …
      Tensor shape : (4, 300, 8, 1)

[3/4] Loading CTR-GCN checkpoint …
  Checkpoint epoch : 87
  Val acc at save  : 72.50%

[4/4] Running inference …

──────────────────────────────────────────────────────────
  RESULT
──────────────────────────────────────────────────────────
  Predicted exercise : SHOULDER_ROTATION
  Confidence         : 81.34%
  Inference time     : 12.4 ms

  All class probabilities:
   1. shoulder_rotation      81.34%  ████████████████ ◄
   2. shoulder_flexion       11.20%  ██
   3. shoulder                5.92%  █
   4. elbow                   1.54%
──────────────────────────────────────────────────────────
```

---

## Prerequisites

The model checkpoint must exist before running inference:

```bash
# Train first if checkpoint doesn't exist
python training/train_upper_limb_ctrgcn.py
```

---

## Supported Input Formats

`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.wmv`
