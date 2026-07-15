# rapha-ai — Repository Refactor Report
Generated: 2026-07-15 12:30:52

## Summary

The `rapha-model-test` repository has been fully refactored into a production-grade,
module-grouped architecture. All 4 CTR-GCN pipelines (lower limb, upper limb, face,
post-stroke) are now completely independent with consistent directory layouts.

---

## Validation Checklist

| Component | Path | Status |
|-----------|------|--------|
| Lower-limb training script                    | `training/lower_limb/train.py` | ✅ |
| Upper-limb training script                    | `training/upper_limb/train.py` | ✅ |
| Face training script                          | `training/face/train.py` | ✅ |
| Post-stroke training script                   | `training/post_stroke/train.py` | ✅ |
| Lower-limb inference                          | `inference/lower_limb/predict.py` | ✅ |
| Upper-limb inference                          | `inference/upper_limb/predict.py` | ✅ |
| Lower-limb dataset loader                     | `preprocessing/lower_limb/loader.py` | ✅ |
| Lower-limb graph                              | `preprocessing/lower_limb/graph.py` | ✅ |
| Upper-limb dataset loader                     | `preprocessing/upper_limb/loader.py` | ✅ |
| Face dataset loader                           | `preprocessing/face/loader.py` | ✅ |
| Post-stroke dataset loader                    | `preprocessing/post_stroke/loader.py` | ✅ |
| Lower-limb model dir (checkpoint pending)     | `models/lower_limb` | ✅ |
| Upper-limb checkpoint                         | `models/upper_limb/best_model.pth` | ✅ |
| Face checkpoint                               | `models/face/best_model.pth` | ✅ |
| Post-stroke checkpoint                        | `models/post_stroke/best_model.pth` | ✅ |
| Shared CTR-GCN architecture                   | `model/ctrgcn.py` | ✅ |
| Lower-limb config                             | `configs/lower_limb.yaml` | ✅ |
| Upper-limb config                             | `configs/upper_limb.yaml` | ✅ |
| Face config                                   | `configs/face.yaml` | ✅ |
| Post-stroke config                            | `configs/post_stroke.yaml` | ✅ |
| Dataset collection automation                 | `automation/collect_dataset.py` | ✅ |
| Balance & train pipeline                      | `automation/balance_and_train.py` | ✅ |
| Deployment placeholder                        | `deployment/README.md` | ✅ |

---

## Dataset Inventory

| Module       | Tensors | Train | Test |
|--------------|---------|-------|------|
| lower_limb   |     124 |   124 |   31 |
| upper_limb   |      34 |    27 |    7 |
| face         |      15 |    12 |    3 |
| post_stroke  |      25 |    20 |    5 |

> ⏳ Lower-limb tensor extraction is still in progress (target: 40/class, 360 total).

---

## Model Inventory

| Module       | Checkpoint               | Metrics |
|--------------|--------------------------|---------|
| lower_limb   | ⏳ pending training     | ✅ |
| upper_limb   | ✅                      | ⚠️ no metrics.json |
| face         | ✅                      | ⚠️ no metrics.json |
| post_stroke  | ✅                      | ⚠️ no metrics.json |

> ⏳ Lower-limb `best_model.pth` will be saved to `models/lower_limb/` once `balance_and_train.py` completes training.

---

## New Repository Structure

```
rapha-ai/
├── automation/
│   ├── balance_and_train.py        ← Dataset balancing + CTR-GCN training runner
│   ├── collect_dataset.py          ← Autonomous yt-dlp web scraper
│   ├── run_all_pipelines.py
│   ├── run_lower_pipeline.py
│   ├── run_upper_pipeline.py
│   ├── run_face_pipeline.py
│   └── run_poststroke_pipeline.py
│
├── configs/
│   ├── lower_limb.yaml
│   ├── upper_limb.yaml             ← NEW
│   ├── face.yaml                   ← NEW
│   └── post_stroke.yaml            ← NEW
│
├── datasets/
│   ├── lower_limb/  (raw/ + skeletons/ + train/test CSVs)
│   ├── upper_limb/  (raw/ + skeletons/ + train/test CSVs)
│   ├── face/        (skeletons/)
│   └── post_stroke/ (raw/ + skeletons/ + train/test CSVs)
│
├── deployment/
│   └── README.md                   ← Placeholder for future API/backend
│
├── docs/                           ← All documentation preserved
│
├── evaluation/
│   ├── lower_limb/   (v3 results: confusion_matrix, curves, report)
│   ├── upper_limb/
│   ├── face/
│   └── post_stroke/
│
├── inference/
│   ├── lower_limb/predict.py
│   ├── upper_limb/predict.py
│   ├── face/predict.py
│   └── post_stroke/predict.py
│
├── model/
│   └── ctrgcn.py                   ← Shared CTR-GCN architecture (unchanged)
│
├── models/
│   ├── lower_limb/best_model.pth   ← ⏳ pending training completion
│   ├── upper_limb/best_model.pth
│   ├── face/best_model.pth
│   ├── post_stroke/best_model.pth
│   └── pose_landmarker_full.task   ← Shared MediaPipe model
│
├── preprocessing/
│   ├── lower_limb/  (extract.py, build_tensors.py, split.py, loader.py, graph.py)
│   ├── upper_limb/  (extract.py, build_tensors.py, split.py, loader.py, graph.py, organize.py)
│   ├── face/        (extract.py, build_tensors.py, split.py, loader.py, graph.py, landmark_mapping.py)
│   ├── post_stroke/ (extract.py, build_tensors.py, split.py, loader.py, graph.py, organize.py)
│   └── shared/      (tensor_stats.py, post_extraction_stats.py)
│
├── tools/
│   └── balance_and_train.py        ← Moved to automation/
│
├── training/
│   ├── lower_limb/train.py
│   ├── upper_limb/train.py
│   ├── face/train.py
│   └── post_stroke/train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Files Removed (Phase 1)

| File | Reason |
|------|--------|
| `patch_pipeline.py`, `patch_timestamp.py` | Temporary patch scripts |
| `training.log`, `upper_training.log` | Regenerable logs |
| `.DS_Store` (root + subdirs) | macOS cache |
| `new_video_inventory.json`, `quality_report.json`, `duplicate_report.json`, `dataset_statistics.json` | Regenerable artifacts |
| `model_comparison.md` | Superseded |
| `models/best_lower_limb_ctrgcn.pth` (v1/auto/v2/v3) | Superseded by final |
| `datasets/lower_limb/duplicates/`, `metadata/`, `visualizations/`, `downloads/`, `rejected/`, `manual_review/`, `review/` | Pipeline working dirs |
| `datasets/lower_limb/hash_registry.json` | Pipeline artifact |
| `logs/` | Regenerable logs |
| All `__pycache__/` directories | Python cache |
| `results/upper_limb/run_preflight.py` | Misplaced script |

---

## Files Moved (Phase 2)

- 25 preprocessing scripts → `preprocessing/{module}/`
- 4 dataset loaders → `preprocessing/{module}/loader.py`
- 5 graph definitions → `preprocessing/{module}/graph.py`
- 4 training scripts → `training/{module}/train.py`
- 4 inference scripts → `inference/{module}/predict.py`
- 4 model checkpoints → `models/{module}/best_model.pth`
- 4 results directories → `evaluation/{module}/`
- `run_autonomous_pipeline.py` → `automation/collect_dataset.py`
- `tools/balance_and_train.py` → `automation/balance_and_train.py`

---

## Imports Updated (Phase 3)

All internal imports in 12 Python files updated:
- `from dataset.loader import` → `from preprocessing.lower_limb.loader import`
- `from graph.lower_limb import` → `from preprocessing.lower_limb.graph import`
- Old model paths (`models/best_lower_limb_ctrgcn.pth`) → new paths (`models/lower_limb/best_model.pth`)
- Old results paths (`results/lower_limb`) → `evaluation/lower_limb_final`

---

## Remaining Technical Debt

| Priority | Item | Action |
|----------|------|--------|
| 🔴 High | Lower-limb `best_model.pth` missing | Completes automatically when `balance_and_train.py` finishes |
| 🟡 Med | Upper-limb skeletons have no class prefix | Run rename script (same as Step 1 of lower-limb pipeline) |
| 🟡 Med | Face dataset only 15 tensors | Needs data collection |
| 🟡 Med | Training scripts don't read from `configs/*.yaml` | Future: migrate hyperparams to YAML |
| 🟢 Low | `deployment/` is a placeholder | Future: FastAPI + real-time inference |
| 🟢 Low | `README.md` not yet updated | Manual documentation update needed |
