# Post-Stroke Dataset Creation

## Overview

This document describes how the post-stroke rehabilitation video dataset is organised and processed into CTR-GCN-compatible tensors.

---

## Step 1 — Organise Raw Videos

Raw videos are copied from the raw downloads directory and classified by filename analysis into `dataset_raw_poststroke/`.

```bash
python preprocessing/organize_poststroke_dataset.py
```

### Classification Rules

| Keyword(s) in filename | → Class |
|---|---|
| `reaching` | `reaching` |
| `crumling`, `grasp` | `grasp_release` |
| `shoulder-flexio`, `shoulder flexion` | `shoulder_flexion` |
| `horizontal range of motion` | `shoulder_abduction` |
| `elbow-flexion`, `elbow flexion`, `elbow` | `elbow_flexion` |
| `sit-to-stand`, `getting up`, `squat` | `sit_to_stand` |
| `weight-shift`, `weight shift`, `weight shifts` | `weight_shift` |
| `heel`, `lunge`, `knee-flexion`, `calf`, `crossing` | `gait_training` |
| `stance`, `standing-feet`, `eyes-open-closed` | `balance_training` |
| `trunk`, `side-to-side` | `trunk_rotation` |
| No match | `unclassified` |

### Outputs

```
dataset_raw_poststroke/
    reaching/
    grasp_release/
    shoulder_flexion/
    shoulder_abduction/
    elbow_flexion/
    sit_to_stand/
    weight_shift/
    gait_training/
    balance_training/
    trunk_rotation/
    unclassified/
    classification_report.csv        ← per-video classification log
```

---

## Step 2 — Extract Landmarks with MediaPipe Holistic

```bash
python preprocessing/extract_poststroke_dataset.py
```

Runs MediaPipe Holistic in video mode on every frame of every classified video to capture joint landmarks.

### Joint Mapping (V=12)

| MediaPipe Landmark | Graph Node | Meaning |
|---|---|---|
| 11 | 0 | Left Shoulder |
| 12 | 1 | Right Shoulder |
| 13 | 2 | Left Elbow |
| 14 | 3 | Right Elbow |
| 15 | 4 | Left Wrist |
| 16 | 5 | Right Wrist |
| 23 | 6 | Left Hip |
| 24 | 7 | Right Hip |
| 25 | 8 | Left Knee |
| 26 | 9 | Right Knee |
| 27 | 10 | Left Ankle |
| 28 | 11 | Right Ankle |

Missing landmarks are filled with `-1` (sentinel). No frames are skipped.

### Output

```
processed_dataset_poststroke/poststroke_frame_labels.csv
```

Columns: `video_name, frame, label, joint_0_x, joint_0_y, joint_0_z, joint_0_visibility, … joint_11_visibility`

---

## Step 3 — Build CTR-GCN Tensors

```bash
python preprocessing/build_poststroke_ctrgcn_dataset.py
```

Converts the frame-level CSV into fixed-length skeleton tensors of shape `(C, T, V, M)` where:
- $C = 4$ (x, y, z, visibility)
- $T = 300$ (normalized frames)
- $V = 12$ (joints)
- $M = 1$ (person)

Padding (ceiling repeating) is applied if $T < 300$, and uniform linspace sampling is applied if $T > 300$.

### Output

```
processed_dataset_poststroke/skeletons/*.npy   ← Npy tensors
```

---

## Step 4 — Train/Test Dataset Split

```bash
python preprocessing/split_poststroke_dataset.py
```

Performs an 80/20 stratified train/test split. Generates:
- `processed_dataset_poststroke/train_labels.csv`
- `processed_dataset_poststroke/test_labels.csv`
- `processed_dataset_poststroke/poststroke_class_map.csv`
