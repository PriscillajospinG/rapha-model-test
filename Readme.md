# Rapha — Physiotherapy Rehabilitation AI Platform

> **Research-grade skeleton action recognition for physiotherapy exercise classification using MediaPipe + CTR-GCN.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2F12.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Research-lightgrey)](#)

---

## Overview

Rapha is a multi-domain physiotherapy rehabilitation platform that classifies rehabilitation exercises from video using:

- **MediaPipe Pose** — landmark extraction (33 body keypoints)
- **CTR-GCN** — Channel-wise Topology Refinement Graph Convolutional Network
- **PyTorch** — training framework with NVIDIA AMP support

### Rehabilitation Domains

| Domain | Classes | Status |
|---|---|---|
| **Lower Limb** | ankle, calf, hamstring, heel_slide, hip_abduction, knee_extension, leg_raise, quadriceps_set, toe_raise | ✅ Active |
| **Upper Limb** | shoulder, elbow, wrist, … | ✅ Active |
| **Facial** | facial_exercises, … | ✅ Active |
| **Post-Stroke** | gait, balance, arm_recovery, … | 🚧 Coming |

---

## Repository Structure

```
rapha-model-test/
├── configs/                    # Centralized YAML configurations
│   └── lower_limb.yaml         # All hyperparams, paths, class map
│
├── dataset/                    # Domain-organized datasets (not in git)
│   └── lower_limb/
│       ├── raw/                # Raw videos by class (not in git)
│       ├── processed/          # Extracted tensors + split CSVs (not in git)
│       └── metadata/           # class_map.json, split_manifest.json ✅ in git
│
├── dataset_versions/           # Version snapshots (lightweight, ✅ in git)
│   ├── lower_limb_v1/
│   └── latest -> lower_limb_v1
│
├── experiments/                # Training experiments (configs + metrics ✅ in git)
│   └── lower_limb/
│       └── experiment_001/
│
├── src/
│   ├── preprocessing/          # Extraction, tensor building, splitting, validation
│   ├── loaders/                # PyTorch Dataset classes
│   └── training/               # CTR-GCN training scripts
│
├── automation/
│   └── run_lower_pipeline.py   # 🚀 One-command pipeline runner
│
├── tools/
│   ├── classify_new_videos.py  # Map new external videos to class folders
│   ├── migrate_repo.py         # One-time migration from old structure
│   └── version_dataset.py      # Snapshot dataset as a new version
│
├── graph/                      # Skeleton graph topology definitions
├── model/                      # CTR-GCN model architecture
└── inference/                  # Inference / prediction scripts
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# For NVIDIA GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Add your videos

Place raw videos into class subfolders:
```
dataset/lower_limb/raw/
    ankle/
    calf/
    hamstring/
    heel_slide/
    hip_abduction/
    knee_extension/
    leg_raise/
    quadriceps_set/
    toe_raise/
```

Or use the classifier for unorganized video collections:
```bash
python tools/classify_new_videos.py --source "/path/to/videos" --dry-run
# Review tools/proposed_classification.json, then:
python tools/classify_new_videos.py --confirm
```

### 3. Run the full pipeline

```bash
python automation/run_lower_pipeline.py
```

That's it. The pipeline:
1. Detects only NEW videos (skips already-processed ones)
2. Runs MediaPipe extraction
3. Builds CTR-GCN tensors
4. Generates stratified 70/15/15 train/val/test split
5. Validates dataset integrity (8 checks)
6. Generates class distribution plots
7. Trains CTR-GCN with AMP
8. Saves checkpoint + experiment artifacts
9. Versions the dataset snapshot

### 4. Add more videos later

Drop new videos into the class folders and run:
```bash
python automation/run_lower_pipeline.py
```
Only the new videos are processed. Existing tensors are untouched.

---

## Pipeline Flags

```bash
# Skip extraction (tensors already built), run validation + training only
python automation/run_lower_pipeline.py --skip-extraction

# Build tensors only, skip training
python automation/run_lower_pipeline.py --skip-training

# Jump directly to training (tensors must exist)
python automation/run_lower_pipeline.py --only-train

# Force full re-extraction and tensor rebuild from scratch
python automation/run_lower_pipeline.py --full-rebuild

# Custom experiment name
python automation/run_lower_pipeline.py --experiment-name ablation_aug
```

---

## Dataset Versioning

Every training run creates a versioned snapshot:
```bash
dataset_versions/
├── lower_limb_v1/
│   ├── manifest.json           # video list, hashes, split, git hash
│   ├── class_map.json
│   └── dataset_statistics.json
└── latest -> lower_limb_v1    # symlink to current version
```

List all versions:
```bash
python tools/version_dataset.py --list
```

---

## Experiment Tracking

Each training run auto-creates a numbered experiment directory:
```
experiments/lower_limb/
└── experiment_001/
    ├── config.yaml             # exact hyperparameters used
    ├── metrics.json            # accuracy, git hash, dataset version, timestamp
    ├── best_model.pth          # best validation checkpoint (not in git)
    ├── confusion_matrix.png
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── classification_report.txt
```

---

## NVIDIA GPU Compatibility

| GPU | Status |
|---|---|
| NVIDIA A100 (80GB SXM) | ✅ Tested |
| NVIDIA H100 | ✅ Compatible |
| NVIDIA L40S | ✅ Compatible |
| NVIDIA L4 | ✅ Compatible |
| NVIDIA T4 | ✅ Compatible |

Features:
- **AMP mixed precision** — `torch.cuda.amp.autocast()` + `GradScaler`
- **cudnn.benchmark** — optimal kernel selection for fixed-size inputs
- **pin_memory** — zero-copy CPU→GPU data transfer
- **persistent_workers** — avoid DataLoader worker restart overhead
- **Auto num_workers** — `min(8, cpu_count)` on CUDA, 0 on CPU

---

## Reproducibility

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for exact commands to reproduce any experiment.

---

## Dataset Download

Raw videos are NOT stored in this repository. See [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md) for instructions on:
- Where to source physiotherapy exercise videos
- How to organize them into class folders
- How to validate your local dataset against a published manifest

---

## Citation

If you use this repository in your research, please cite:
```bibtex
@software{rapha2026,
  title  = {Rapha: Physiotherapy Rehabilitation AI Platform},
  year   = {2026},
  url    = {https://github.com/your-org/rapha-model}
}
```

---

## License

This repository is released for research use. See LICENSE for details.
