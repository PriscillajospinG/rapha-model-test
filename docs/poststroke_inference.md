# Post-Stroke CTR-GCN Inference

## Overview

Run the trained model on any new post-stroke rehabilitation video to get an exercise prediction with confidence scores.

---

## Usage

```bash
python inference/predict_poststroke_video.py --video path/to/exercise.mp4
```

### Optional: specify a custom checkpoint

```bash
python inference/predict_poststroke_video.py \
    --video  path/to/exercise.mp4 \
    --model  models/best_poststroke_ctrgcn.pth
```

---

## Inference Pipeline

```
Input Video
    ↓
[1/4] MediaPipe Holistic — extracts pose joints 11-16, 23-28
    ↓
[2/4] Build GCN tensor (C=4, T=300, V=12, M=1)
      • frames > 300 → uniform temporal sampling
      • frames < 300 → tile/loop to reach 300
    ↓
[3/4] Load best_poststroke_ctrgcn.pth
    ↓
[4/4] CTR-GCN forward pass → softmax probabilities
    ↓
Display: Predicted class + confidence + ranked probabilities
```

---

## Example Output

```
============================================================
  Post-Stroke CTR-GCN Video Inference
============================================================
  Device    : cpu
  Video path: /Users/priscillajosping/Downloads/Post Stroke Excercises/Sit-to-Stand(Post-Stroke-Exercise).mp4
  Model path: /Users/priscillajosping/Downloads/CV_dev/models/best_poststroke_ctrgcn.pth

[1/4] Extracting landmarks via MediaPipe Holistic …
  Video : Sit-to-Stand(Post-Stroke-Exercise).mp4
  Frames: 240  |  FPS: 30.0
  Frames extracted : 240
  Failed detections: 0 (0.0%)

[2/4] Constructing input GCN skeleton tensor …
  Tensor shape: (4, 300, 12, 1) (C, T, V, M)

[3/4] Loading model checkpoint …
  Checkpoint epoch : 100
  Val acc at save  : 100.00%
  Classes          : {0: 'balance_training', 1: 'elbow_flexion', 2: 'gait_training', 3: 'grasp_release', 4: 'reaching', 5: 'shoulder_abduction', 6: 'shoulder_flexion', 7: 'sit_to_stand', 8: 'trunk_rotation', 9: 'weight_shift'}

[4/4] Running forward pass through CTR-GCN …

============================================================
  INFERENCE RESULTS
============================================================
  Predicted Exercise : sit_to_stand
  Confidence         : 98.42%

  Class Probabilities:
    - sit_to_stand              :  98.42%
    - weight_shift              :   1.12%
    - gait_training             :   0.31%
    - balance_training          :   0.15%
    - reaching                  :   0.00%
    - grasp_release             :   0.00%
    - shoulder_flexion          :   0.00%
    - shoulder_abduction        :   0.00%
    - elbow_flexion             :   0.00%
    - trunk_rotation            :   0.00%
------------------------------------------------------------
  Extraction time: 5.43 s
  Inference time : 14.2 ms
============================================================
```

---

## Prerequisites

The model checkpoint must exist before running inference:

```bash
# Train first if checkpoint doesn't exist
python training/train_poststroke_ctrgcn.py
```
