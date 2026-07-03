# Face Dataset Creation Guide

This document explains how to label raw facial rehabilitation videos and build
the CTR-GCN skeleton tensors used for training.

---

## Prerequisites

```bash
pip install opencv-python mediapipe pandas tqdm scikit-learn
```

Place raw face videos in the following structure:

```
dataset_raw_face/
├── Eyebrows/
│   ├── video1.mp4
│   └── ...
├── Eyes/
├── Frown/
├── Lips/
├── Nose/
└── Side Chicks/
```

> The `dataset_raw_face/` directory is a symlink pointing to the actual raw
> dataset folder. Create it with:
> ```bash
> ln -sf "/path/to/face raw" dataset_raw_face
> ```

---

## Step 1 — Frame-level landmark extraction

```bash
python preprocessing/extract_face_dataset.py
```

**What it does:**
- Scans every video under `dataset_raw_face/`
- Runs MediaPipe FaceMesh on every frame
- Extracts 33 physiotherapy-specific facial landmarks per frame
- Saves `x`, `y`, `z` coordinates for each landmark (no visibility)
- On failed detection: fills all values with `-1` (no frames are skipped)
- Outputs: `processed_dataset_face/face_frame_labels.csv`
- Log: `face_extraction.log`

**Landmark selection (33 nodes):**

| Region | MediaPipe IDs | Node indices |
|---|---|---|
| Left Eyebrow | 70, 63, 105, 66, 107 | 0–4 |
| Right Eyebrow | 336, 296, 334, 293, 300 | 5–9 |
| Eyes | 33, 133, 362, 263, 159, 145, 386, 374 | 10–17 |
| Cheeks | 50, 280, 187, 411 | 18–21 |
| Nose | 1, 4, 168 | 22–24 |
| Mouth | 61, 291, 13, 14, 78, 308, 17, 0 | 25–32 |

**CSV columns:**

```
video_name, frame, label,
landmark_0_x, landmark_0_y, landmark_0_z,
landmark_1_x, landmark_1_y, landmark_1_z,
...
landmark_32_x, landmark_32_y, landmark_32_z
```

---

## Step 2 — Build CTR-GCN tensors

```bash
python preprocessing/build_face_ctrgcn_dataset.py
```

**What it does:**
- Reads `face_frame_labels.csv`
- Groups rows by `video_name`
- Replaces `-1` sentinel values with `0.0`
- Resamples each video to exactly **T=300 frames**:
  - If `T_raw > 300`: uniform subsampling
  - If `T_raw < 300`: tiling (repeat and truncate)
- Transposes to shape `(C, T, V, M) = (3, 300, 33, 1)`
- Saves each video as a `.npy` file

**Output:** `processed_dataset_face/skeletons/<sample_name>.npy`

Each file has shape `(3, 300, 33, 1)` and dtype `float32`.

---

## Step 3 — Train / test split

```bash
python preprocessing/split_face_dataset.py
```

**What it does:**
- Scans `processed_dataset_face/skeletons/` for `.npy` files
- Matches each file to its class label via the frame CSV
- Builds a dynamic integer class map (sorted alphabetically for reproducibility)
- Applies **80/20 stratified** split (falls back to random if any class has only 1 sample)

**Outputs:**
- `processed_dataset_face/train_labels.csv`
- `processed_dataset_face/test_labels.csv`
- `processed_dataset_face/face_class_map.csv`

---

## Dataset Statistics

| Property | Value |
|---|---|
| Classes | 6 (Eyebrows, Eyes, Frown, Lips, Nose, Side Chicks) |
| Total videos | 15 |
| Tensor shape | (3, 300, 33, 1) |
| Channels | x, y, z (no visibility) |
| Frames per sample | 300 (standardised) |
| Train split | ~80% |
| Test split | ~20% |
