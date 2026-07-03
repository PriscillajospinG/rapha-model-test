# Dataset Creation & Preparation Guide

This document describes the pipeline for preparing the lower-limb CTR-GCN physiotherapy dataset from raw video files.

---

## 1. Raw Dataset structure

Raw physiotherapy video recordings should be organized by exercise category under `dataset_raw/`:

```text
dataset_raw/
├── ankle/
│   ├── video1.mp4
│   └── video2.mp4
├── calf/
├── hamstring/
├── heel_slide/
├── hip/
├── knee/
├── leg_raise/
├── quadriceps/
└── toes/
```

Any video files (`.mp4`, `.avi`, `.mov`, etc.) placed in these directories are automatically associated with the respective folder name as their class label.

---

## 2. Landmark Extraction (MediaPipe Pose)

To scan the raw video files and extract skeleton landmark coordinates frame-by-frame, run:

```bash
python preprocessing/extract_lower_limb_dataset.py
```

### Steps Performed:
1. **Video Ingestion**: Recursively crawls `dataset_raw/` for supported video formats.
2. **Pose Detection**: Runs MediaPipe Pose (Full Model complexity = 1) in tracking mode on every frame.
3. **Joint Filtering**: Isolates and extracts only the 10 lower-limb joints:
   - `23`: `LEFT_HIP`
   - `24`: `RIGHT_HIP`
   - `25`: `LEFT_KNEE`
   - `26`: `RIGHT_KNEE`
   - `27`: `LEFT_ANKLE`
   - `28`: `RIGHT_ANKLE`
   - `29`: `LEFT_HEEL`
   - `30`: `RIGHT_HEEL`
   - `31`: `LEFT_FOOT_INDEX`
   - `32`: `RIGHT_FOOT_INDEX`
4. **Data Logging**: Records four attributes per joint: `x`, `y`, `z`, and `visibility`. Missing frames or failed pose detections are assigned a sentinel value of `-1`.
5. **Output**: Writes frame-level logs directly to `processed_dataset/lower_limb_frame_labels.csv`.

---

## 3. CTR-GCN Tensor Building

Convert the frame-level CSV landmarks into fixed-length 4D NumPy arrays by running:

```bash
python preprocessing/build_ctrgcn_dataset.py
```

### Conversion logic:
- **Grouping**: Groups the CSV entries by individual `video_name`.
- **Temporal Resampling**:
  - For sequences with length $T_{raw} \ge 300$, uniformly samples exactly $300$ frames.
  - For shorter sequences ($T_{raw} < 300$), loops/tiles the frames to pad the sequence up to $300$ frames.
- **Sentinel Handling**: Converts missing joint sentinel values (`-1`) to `0.0` for numerical training stability.
- **Reshaping**: Transforms the array structure from $(T_{raw}, V, C)$ to the format required by the CTR-GCN architecture:
  $$\text{Shape: } (C, T, V, M) \implies (4, 300, 10, 1)$$
  Where:
  - $C = 4$ channels ($x$, $y$, $z$, and $visibility$)
  - $T = 300$ frames
  - $V = 10$ joints (re-mapped to $0$–$9$)
  - $M = 1$ person
- **Output**: Saves each sample as an independent `.npy` file under `processed_dataset/skeletons/`.

---

## 4. Train/Test Dataset Splitting

Generate the train and test dataset splits by running:

```bash
python preprocessing/split_dataset.py
```

### Splitting Process:
1. Scans `processed_dataset/skeletons/` to find all generated `.npy` files.
2. Derives the class label from the filename prefix (e.g. `knee_Knee_Hold.npy` maps to class index `7` representing `knee`).
3. Splits the dataset into an **80/20 train/test split**.
4. Applies stratification to preserve the proportion of each class in both sets.
5. **Outputs**: Saves the split metadata in `processed_dataset/train_labels.csv` and `processed_dataset/test_labels.csv`.
