# Upper-Limb Dataset Creation

## Overview

This document describes how the upper-limb physiotherapy video dataset is organised and processed into CTR-GCN-compatible tensors.

---

## Step 1 — Organise Raw Videos

Raw videos are located in an unorganised directory and are automatically classified by filename analysis.

```bash
python preprocessing/organize_upper_dataset.py
```

### Classification Rules (priority order)

| Keyword(s) in filename | → Class |
|---|---|
| `wrist` | `wrist` |
| `bicep`, `curl` | `bicep_curl` |
| `tricep` | `tricep_extension` |
| `arm raise`, `lateral raise` | `arm_raise` |
| `abduction` | `shoulder_abduction` |
| `shoulder flexion`, `assisted flexion` | `shoulder_flexion` |
| `rotation` (any form) | `shoulder_rotation` |
| `elbow` | `elbow` |
| `shoulder`, `scapular`, `serratus`, etc. | `shoulder` |
| No match | `unclassified` |

### Outputs

```
dataset_raw_upper/
    shoulder/
    shoulder_flexion/
    shoulder_rotation/
    elbow/
    wrist/
    arm_raise/
    shoulder_abduction/
    bicep_curl/
    tricep_extension/
    unclassified/
    classification_report.csv        ← per-video classification log
```

The `classification_report.csv` has columns:

| Column | Description |
|---|---|
| `video_name` | Original filename |
| `source_path` | Relative path inside the raw source |
| `predicted_class` | Assigned exercise class |
| `confidence` | Classification confidence (0–1) |
| `reason` | Human-readable rule that triggered |

---

## Step 2 — Extract Landmarks with MediaPipe

```bash
python preprocessing/extract_upper_limb_dataset.py
```

Runs MediaPipe Pose in video mode on every frame of every classified video.

### Joint Mapping

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

Missing landmarks are filled with `-1` (sentinel). No frames are skipped.

### Output

```
processed_dataset_upper/upper_limb_frame_labels.csv
```

Columns: `video_name, frame, label, joint_0_x, joint_0_y, joint_0_z, joint_0_visibility, … joint_7_visibility`

---

## Step 3 — Build CTR-GCN Tensors

```bash
python preprocessing/build_upper_ctrgcn_dataset.py
```

Converts the frame-level CSV into fixed-length tensors:

| Dimension | Value | Meaning |
|---|---|---|
| C | 4 | x, y, z, visibility |
| T | 300 | standardised frame count |
| V | 8 | upper-limb graph nodes |
| M | 1 | single person |

**Resampling rules:**
- `frames > 300` → uniform temporal sampling
- `frames < 300` → tile (loop) the sequence to reach 300

### Output

```
processed_dataset_upper/skeletons/<sample_name>.npy
```

Each file: `shape = (4, 300, 8, 1)`, `dtype = float32`

---

## Step 4 — Train / Test Split

```bash
python preprocessing/split_upper_dataset.py
```

Applies a stratified 80/20 split. The class map is built **dynamically** from whatever classes are present in the data.

### Outputs

```
processed_dataset_upper/train_labels.csv
processed_dataset_upper/test_labels.csv
processed_dataset_upper/upper_class_map.csv   ← class_name → integer label
```
