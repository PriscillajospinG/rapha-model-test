# Reproducibility Guide
## Exact Commands to Reproduce Any Experiment

---

## Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/rapha-model-test.git
cd rapha-model-test

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# For NVIDIA GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Install PyYAML (required for config loading)
pip install pyyaml
```

---

## Reproducing a Specific Experiment

Every experiment directory contains a complete record:
```
experiments/lower_limb/experiment_001/
├── config.yaml              ← ALL hyperparameters used
├── metrics.json             ← git hash, dataset version, accuracy
├── classification_report.txt
└── confusion_matrix.png
```

### Step 1 — Check the git commit

```bash
# See what commit produced a result
cat experiments/lower_limb/experiment_001/metrics.json | python -m json.tool | grep git
```

```json
"git_commit": "a3f9b12",
"git_branch": "main"
```

```bash
# Checkout that exact commit
git checkout a3f9b12
```

### Step 2 — Identify the dataset version

```bash
cat experiments/lower_limb/experiment_001/metrics.json | python -m json.tool | grep dataset_version
```

```json
"dataset_version": "lower_limb_v2"
```

```bash
# See the exact video list for that version
cat dataset_versions/lower_limb_v2/manifest.json
```

### Step 3 — Obtain the dataset

Place all videos from the manifest's `samples` list into `dataset/lower_limb/raw/<class>/`.

### Step 4 — Reproduce exactly

```bash
# Run only training (skip extraction if tensors exist)
python automation/run_lower_pipeline.py --skip-extraction \
    --experiment-name reproduce_experiment_001
```

Or full pipeline from raw videos:
```bash
python automation/run_lower_pipeline.py \
    --full-rebuild \
    --experiment-name reproduce_experiment_001
```

---

## Configuration

All hyperparameters are in `configs/lower_limb.yaml`. The experiment's `config.yaml` is an exact snapshot of this file at training time.

Key parameters:
```yaml
training:
  epochs:        100
  batch_size:    8
  learning_rate: 0.001
  weight_decay:  0.0001
  seed:          42
  label_smoothing: 0.1
  scheduler:     cosine

split:
  train: 0.70
  val:   0.15
  test:  0.15
  seed:  42
  stratify: true
```

---

## Validating Reproduction

```bash
# Check test accuracy matches
python -c "
import json
with open('experiments/lower_limb/experiment_001/metrics.json') as f:
    orig = json.load(f)
with open('experiments/lower_limb/reproduce_experiment_001/metrics.json') as f:
    repro = json.load(f)
print(f'Original:     {orig[\"final_test_acc\"]:.2f}%')
print(f'Reproduced:   {repro[\"final_test_acc\"]:.2f}%')
print(f'Difference:   {abs(orig[\"final_test_acc\"] - repro[\"final_test_acc\"]):.2f}%')
"
```

> **Note:** Minor differences (<1%) are expected due to CUDA non-determinism unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set.

---

## Running Individual Pipeline Steps

```bash
# Step by step (useful for debugging)

# 1. Inventory
python src/preprocessing/video_inventory.py

# 2. Extract (incremental)
python src/preprocessing/extract_lower_limb.py

# 3. Build tensors (incremental)
python src/preprocessing/build_lower_limb.py

# 4. Split
python src/preprocessing/split_lower_limb.py

# 5. Validate
python src/preprocessing/validate_dataset.py

# 6. Statistics
python src/preprocessing/generate_statistics.py

# 7. Train
python src/training/train_lower_limb.py

# 8. Version
python tools/version_dataset.py
```

---

## Common Issues

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce `batch_size` in `configs/lower_limb.yaml` |
| `FileNotFoundError: *.npy` | Run tensor building step first |
| `Tensor shape mismatch` | Delete `tensors/` and rebuild: `--full-rebuild` |
| `No valid rows in CSV` | Re-run the split step |
| `Cannot open video` | Check for corrupted files: `python src/preprocessing/validate_dataset.py` |
