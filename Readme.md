# AI Physiotherapy CTR-GCN

Skeleton-based physiotherapy exercise recognition using Channel-wise Topology
Refinement Graph Convolutional Networks (CTR-GCN) with MediaPipe pose extraction.

Three body regions and one specialized condition are modelled independently:

| Pipeline    | Body Region      | Dataset Directory        |
|-------------|------------------|--------------------------|
| Lower Limb  | Knee / Hip / Ankle / Leg | `processed_dataset/`      |
| Upper Limb  | Shoulder / Elbow / Wrist | `processed_dataset_upper/` |
| Face        | Facial landmarks  | `processed_dataset_face/` |
| Post-Stroke | Post-Stroke Full Body | `processed_dataset_poststroke/` |

---

## One-Command Automation

> **New users: start here.**  A single command executes the complete
> extraction → build → split → training pipeline for any body region.

### Prerequisites

```bash
pip install -r requirements.txt
```

---

### Lower Limb Pipeline

```bash
python automation/run_lower_pipeline.py
```

Runs automatically:

1. `preprocessing/extract_lower_limb_dataset.py`
2. `preprocessing/build_ctrgcn_dataset.py`
3. `preprocessing/split_dataset.py`
4. `training/train_lower_limb_ctrgcn.py`

Log → `logs/lower_pipeline.log`

---

### Upper Limb Pipeline

```bash
python automation/run_upper_pipeline.py
```

Runs automatically:

1. `preprocessing/extract_upper_limb_dataset.py`
2. `preprocessing/build_upper_ctrgcn_dataset.py`
3. `preprocessing/split_upper_dataset.py`
4. `training/train_upper_limb_ctrgcn.py`

Log → `logs/upper_pipeline.log`

---

### Face Pipeline

```bash
python automation/run_face_pipeline.py
```

Runs automatically:

1. `preprocessing/extract_face_dataset.py`
2. `preprocessing/build_face_ctrgcn_dataset.py`
3. `preprocessing/split_face_dataset.py`
4. `training/train_face_ctrgcn.py`

Log → `logs/face_pipeline.log`

---

### Post-Stroke Pipeline

```bash
python automation/run_poststroke_pipeline.py
```

Runs automatically:

1. `preprocessing/extract_poststroke_dataset.py`
2. `preprocessing/build_poststroke_ctrgcn_dataset.py`
3. `preprocessing/split_poststroke_dataset.py`
4. `training/train_poststroke_ctrgcn.py`

Log → `logs/poststroke_pipeline.log`

---

### Master Menu (all pipelines)

```bash
python automation/run_all_pipelines.py
```

Presents an interactive menu:

```
══════════════════════════════════════════════════════════
  AI PHYSIOTHERAPY — CTR-GCN AUTOMATION SYSTEM
══════════════════════════════════════════════════════════

  [1]  🦵  Run Lower Limb Pipeline
  [2]  💪  Run Upper Limb Pipeline
  [3]  😊  Run Face Pipeline
  [4]  🧠  Run Post-Stroke Pipeline
  [5]  🚀  Run ALL Pipelines  (full project)
  [6]  ⏻   Exit
```

Non-interactive usage (CI / scripting):

```bash
python automation/run_all_pipelines.py --run 5   # run all four
python automation/run_all_pipelines.py --run 1   # lower limb only
python automation/run_all_pipelines.py --run 2   # upper limb only
python automation/run_all_pipelines.py --run 3   # face only
python automation/run_all_pipelines.py --run 4   # post-stroke only
```

---

## Optional Flags

All pipeline scripts accept the following mutually-exclusive flags:

| Flag               | Effect                                                   |
|--------------------|----------------------------------------------------------|
| `--skip-extraction`| Skip steps 1-3; jump to training (skeletons must exist). |
| `--skip-training`  | Run extraction / build / split; skip training.           |
| `--only-train`     | Same as `--skip-extraction` — alias for clarity.         |
| `--only-inference` | Placeholder (not yet implemented).                       |

Examples:

```bash
# Skip re-extraction (skeletons already built)
python automation/run_lower_pipeline.py --skip-extraction

# Only retrain the upper limb model
python automation/run_upper_pipeline.py --only-train

# Run all pipelines in training-only mode
python automation/run_all_pipelines.py --run 5 --only-train
```

---

## Project Structure

```
CV_dev/
│
├── automation/                  ← automation layer
│   ├── __init__.py
│   ├── utils.py                 ← shared helpers
│   ├── run_lower_pipeline.py    ← lower limb end-to-end
│   ├── run_upper_pipeline.py    ← upper limb end-to-end
│   ├── run_face_pipeline.py     ← face end-to-end
│   ├── run_poststroke_pipeline.py← post-stroke end-to-end
│   └── run_all_pipelines.py     ← master menu / CLI
│
├── preprocessing/               ← extraction & build scripts
├── training/                    ← training scripts
├── dataset/                     ← data loaders
├── model/                       ← CTR-GCN model definition
├── graph/                       ← graph topology definitions
├── models/                      ← saved checkpoints (.pth)
├── results/                     ← training plots & outputs
│
├── processed_dataset/           ← lower limb tensors
├── processed_dataset_upper/     ← upper limb tensors
├── processed_dataset_face/      ← face tensors
├── processed_dataset_poststroke/← post-stroke tensors
│
├── dataset_raw/                 ← raw lower limb videos
├── dataset_raw_upper/           ← raw upper limb videos
├── dataset_raw_poststroke/      ← raw post-stroke videos
│
└── logs/                        ← auto-created pipeline logs
    ├── lower_pipeline.log
    ├── upper_pipeline.log
    ├── face_pipeline.log
    ├── poststroke_pipeline.log
    └── master_pipeline.log
```

---

## Labelling Strategy

See [Method.md](Method.md) for the full documentation of the hybrid labelling
approach (Expert Annotation + Weak Supervision + Auto-Labelling).

---

## Error Handling

If a pipeline step fails the automation will:

1. Print `✘ FAILED` with the script path and exit code.
2. Diagnose the error against known patterns (missing CSV, missing `.npy`,
   CUDA OOM, tensor shape mismatch, missing checkpoint, import errors, …).
3. Print a **Suggested Fix** to guide the user.
4. Write the full traceback to the corresponding log file.
5. Abort immediately — no further steps are executed.

---

## Validation

Before each training step the automation checks:

| Pipeline   | Required path                              |
|------------|--------------------------------------------|
| Lower Limb | `processed_dataset/skeletons/*.npy`        |
| Upper Limb | `processed_dataset_upper/skeletons/*.npy`  |
| Face       | `processed_dataset_face/skeletons/*.npy`   |
| Post-Stroke| `processed_dataset_poststroke/skeletons/*.npy`|

If the directory is missing or empty the pipeline aborts with a clear message.
