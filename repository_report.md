# rapha-ai — Repository Refactor Report
Generated: 2026-07-15 12:27:15

## Summary
The repository has been refactored from a flat, script-accumulation layout into a
production-grade, module-grouped architecture supporting 4 independent CTR-GCN pipelines.

## Validation Checklist
| Component | Path | Status |
|-----------|------|--------|
| Lower-limb training script               | `training/lower_limb/train.py` | ✅ |
| Upper-limb training script               | `training/upper_limb/train.py` | ✅ |
| Face training script                     | `training/face/train.py` | ✅ |
| Post-stroke training script              | `training/post_stroke/train.py` | ✅ |
| Lower-limb inference                     | `inference/lower_limb/predict.py` | ✅ |
| Lower-limb dataset loader                | `preprocessing/lower_limb/loader.py` | ✅ |
| Lower-limb graph                         | `preprocessing/lower_limb/graph.py` | ✅ |
| Lower-limb checkpoint                    | `models/lower_limb/best_model.pth` | ❌ MISSING |
| Upper-limb checkpoint                    | `models/upper_limb/best_model.pth` | ✅ |
| Face checkpoint                          | `models/face/best_model.pth` | ✅ |
| Post-stroke checkpoint                   | `models/post_stroke/best_model.pth` | ✅ |
| Shared CTR-GCN architecture              | `model/ctrgcn.py` | ✅ |

## Dataset Inventory
| Module       | Tensors | Train | Test |
|--------------|---------|-------|------|
| lower_limb   |     116 |   124 |   31 |
| upper_limb   |      34 |    27 |    7 |
| face         |      15 |    12 |    3 |
| post_stroke  |      25 |    20 |    5 |

## Model Inventory
| Module       | Checkpoint | Metrics |
|--------------|------------|---------|
| lower_limb   | ❌ | ⚠️ no metrics.json |
| upper_limb   | ✅ | ⚠️ no metrics.json |
| face         | ✅ | ⚠️ no metrics.json |
| post_stroke  | ✅ | ⚠️ no metrics.json |

## Repository Tree
```
rapha-ai/
├── automation
│   ├── __init__.py
│   ├── balance_and_train.py
│   ├── collect_dataset.py
│   ├── run_all_pipelines.py
│   ├── run_face_pipeline.py
│   ├── run_lower_pipeline.py
│   ├── run_poststroke_pipeline.py
│   ├── run_upper_pipeline.py
│   └── utils.py
├── configs
│   ├── face.yaml
│   ├── lower_limb.yaml
│   ├── post_stroke.yaml
│   └── upper_limb.yaml
├── deployment
│   └── README.md
├── docs
│   ├── DATASET_GUIDE.md
│   ├── REPRODUCIBILITY.md
│   ├── dataset_creation.md
│   ├── face_dataset_creation.md
│   ├── face_inference.md
│   ├── face_pipeline_report.md
│   ├── face_training.md
│   ├── final_project_summary.md
│   ├── inference.md
│   ├── nvidia_gpu_compatibility_report.md
│   ├── poststroke_dataset_creation.md
│   ├── poststroke_inference.md
│   ├── poststroke_pipeline_report.md
│   ├── poststroke_training.md
│   ├── repository_audit.md
│   ├── training.md
│   ├── upper_dataset_creation.md
│   ├── upper_inference.md
│   ├── upper_pipeline_report.md
│   └── upper_training.md
├── evaluation
│   ├── face
│   │   ├── accuracy_curve.png
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── loss_curve.png
│   ├── lower_limb
│   │   ├── accuracy_curve.png
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── loss_curve.png
│   │   ├── metrics.json
│   │   └── training_report.md
│   ├── post_stroke
│   │   ├── accuracy_curve.png
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── loss_curve.png
│   └── upper_limb
│       ├── accuracy_curve.png
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── loss_curve.png
│       └── validation_report.txt
├── inference
│   ├── face
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── lower_limb
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── post_stroke
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── upper_limb
│   │   ├── __init__.py
│   │   └── predict.py
│   └── __init__.py
├── model
│   ├── __init__.py
│   └── ctrgcn.py
├── models
│   ├── face
│   │   └── best_model.pth
│   ├── post_stroke
│   │   └── best_model.pth
│   ├── upper_limb
│   │   └── best_model.pth
│   └── pose_landmarker_full.task
├── preprocessing
│   ├── face
│   │   ├── __init__.py
│   │   ├── build_tensors.py
│   │   ├── extract.py
│   │   ├── graph.py
│   │   ├── landmark_mapping.py
│   │   ├── loader.py
│   │   └── split.py
│   ├── lower_limb
│   │   ├── __init__.py
│   │   ├── build_tensors.py
│   │   ├── extract.py
│   │   ├── graph.py
│   │   ├── loader.py
│   │   └── split.py
│   ├── post_stroke
│   │   ├── __init__.py
│   │   ├── build_tensors.py
│   │   ├── extract.py
│   │   ├── graph.py
│   │   ├── loader.py
│   │   ├── organize.py
│   │   └── split.py
│   ├── shared
│   │   ├── __init__.py
│   │   ├── post_extraction_stats.py
│   │   └── tensor_stats.py
│   ├── upper_limb
│   │   ├── __init__.py
│   │   ├── build_tensors.py
│   │   ├── extract.py
│   │   ├── graph.py
│   │   ├── loader.py
│   │   ├── organize.py
│   │   └── split.py
│   └── __init__.py
├── training
│   ├── face
│   │   ├── __init__.py
│   │   └── train.py
│   ├── lower_limb
│   │   ├── __init__.py
│   │   └── train.py
│   ├── post_stroke
│   │   ├── __init__.py
│   │   └── train.py
│   ├── upper_limb
│   │   ├── __init__.py
│   │   └── train.py
│   └── __init__.py
├── Method.md
├── Readme.md
├── datasets/
│   ├── lower_limb/   (raw/ + skeletons/ + CSVs)
│   ├── upper_limb/   (raw/ + skeletons/ + CSVs)
│   ├── face/         (skeletons/)
│   └── post_stroke/  (raw/ + skeletons/ + CSVs)
├── requirements.txt
├── README.md
└── .gitignore
```

## Files Removed
  REMOVED  patch_pipeline.py  (junk/regenerable)
  REMOVED  patch_timestamp.py  (junk/regenerable)
  REMOVED  training.log  (junk/regenerable)
  REMOVED  upper_training.log  (junk/regenerable)
  REMOVED  new_video_inventory.json  (junk/regenerable)
  REMOVED  quality_report.json  (junk/regenerable)
  REMOVED  duplicate_report.json  (junk/regenerable)
  REMOVED  dataset_statistics.json  (junk/regenerable)
  REMOVED  model_comparison.md  (junk/regenerable)
  REMOVED  .DS_Store  (junk/regenerable)
  REMOVED  dataset/.DS_Store  (OS cache)
  REMOVED  tools/__pycache__  (pycache)
  REMOVED  dataset/__pycache__  (pycache)
  REMOVED  training/__pycache__  (pycache)
  REMOVED  graph/__pycache__  (pycache)
  REMOVED  model/__pycache__  (pycache)
  REMOVED  inference/__pycache__  (pycache)
  REMOVED  preprocessing/__pycache__  (pycache)
  REMOVED  automation/__pycache__  (pycache)
  REMOVED  results/upper_limb/__pycache__  (pycache)
  REMOVED  results/upper_limb/run_preflight.py  (misplaced script)
  REMOVED  models/best_lower_limb_ctrgcn.pth  (superseded checkpoint)
  REMOVED  models/best_lower_limb_ctrgcn_auto.pth  (superseded checkpoint)
  REMOVED  models/best_lower_limb_ctrgcn_v2.pth  (superseded checkpoint)
  REMOVED  models/best_lower_limb_ctrgcn_v3.pth  (superseded checkpoint)
  REMOVED  datasets/lower_limb/duplicates  (pipeline working dir)
  REMOVED  datasets/lower_limb/metadata  (pipeline working dir)
  REMOVED  datasets/lower_limb/visualizations  (pipeline working dir)
  REMOVED  datasets/lower_limb/downloads  (pipeline working dir)
  REMOVED  datasets/lower_limb/rejected  (pipeline working dir)
  REMOVED  datasets/lower_limb/manual_review  (pipeline working dir)
  REMOVED  datasets/lower_limb/review  (pipeline working dir)
  REMOVED  datasets/lower_limb/hash_registry.json  (pipeline artifact)
  REMOVED  datasets/lower_limb/lower_limb_frame_labels.csv  (pipeline artifact)
  REMOVED  logs  (pipeline logs)
  REMOVED  dataset  (empty package after migration)
  REMOVED  graph  (empty package after migration)
  REMOVED  results  (migrated to evaluation/)
  REMOVED  tools  (empty after migration)
  CLEAN    preprocessing/ — all scripts migrated

## Files Moved
  REMOVED  patch_pipeline.py  (junk/regenerable)
  REMOVED  patch_timestamp.py  (junk/regenerable)
  REMOVED  training.log  (junk/regenerable)
  REMOVED  upper_training.log  (junk/regenerable)
  REMOVED  new_video_inventory.json  (junk/regenerable)
  REMOVED  quality_report.json  (junk/regenerable)
  REMOVED  duplicate_report.json  (junk/regenerable)
  REMOVED  dataset_statistics.json  (junk/regenerable)
  REMOVED  model_comparison.md  (junk/regenerable)
  REMOVED  .DS_Store  (junk/regenerable)
  REMOVED  dataset/.DS_Store  (OS cache)
  REMOVED  tools/__pycache__  (pycache)
  REMOVED  dataset/__pycache__  (pycache)
  REMOVED  training/__pycache__  (pycache)
  REMOVED  graph/__pycache__  (pycache)
  REMOVED  model/__pycache__  (pycache)
  REMOVED  inference/__pycache__  (pycache)
  REMOVED  preprocessing/__pycache__  (pycache)
  REMOVED  automation/__pycache__  (pycache)
  REMOVED  results/upper_limb/__pycache__  (pycache)
  REMOVED  results/upper_limb/run_preflight.py  (misplaced script)
  REMOVED  models/best_lower_limb_ctrgcn.pth  (superseded checkpoint)
  REMOVED  models/best_lower_limb_ctrgcn_auto.pth  (superseded checkpoint)
  REMOVED  models/best_lower_limb_ctrgcn_v2.pth  (superseded checkpoint)
  REMOVED  models/best_lower_limb_ctrgcn_v3.pth  (superseded checkpoint)
  REMOVED  datasets/lower_limb/duplicates  (pipeline working dir)
  REMOVED  datasets/lower_limb/metadata  (pipeline working dir)
  REMOVED  datasets/lower_limb/visualizations  (pipeline working dir)
  REMOVED  datasets/lower_limb/downloads  (pipeline working dir)
  REMOVED  datasets/lower_limb/rejected  (pipeline working dir)
  REMOVED  datasets/lower_limb/manual_review  (pipeline working dir)
  REMOVED  datasets/lower_limb/review  (pipeline working dir)
  REMOVED  datasets/lower_limb/hash_registry.json  (pipeline artifact)
  REMOVED  datasets/lower_limb/lower_limb_frame_labels.csv  (pipeline artifact)
  REMOVED  logs  (pipeline logs)
  MOVED    preprocessing/extract_lower_limb_dataset.py  →  preprocessing/lower_limb/extract.py  (preprocessing/lower_limb)
  MOVED    preprocessing/build_ctrgcn_dataset.py  →  preprocessing/lower_limb/build_tensors.py  (preprocessing/lower_limb)
  MOVED    preprocessing/split_dataset.py  →  preprocessing/lower_limb/split.py  (preprocessing/lower_limb)
  MOVED    dataset/loader.py  →  preprocessing/lower_limb/loader.py  (preprocessing/lower_limb)
  MOVED    graph/lower_limb.py  →  preprocessing/lower_limb/graph.py  (preprocessing/lower_limb)
  MOVED    preprocessing/extract_upper_limb_dataset.py  →  preprocessing/upper_limb/extract.py  (preprocessing/upper_limb)
  MOVED    preprocessing/build_upper_ctrgcn_dataset.py  →  preprocessing/upper_limb/build_tensors.py  (preprocessing/upper_limb)
  MOVED    preprocessing/split_upper_dataset.py  →  preprocessing/upper_limb/split.py  (preprocessing/upper_limb)
  MOVED    preprocessing/organize_upper_dataset.py  →  preprocessing/upper_limb/organize.py  (preprocessing/upper_limb)
  MOVED    dataset/upper_loader.py  →  preprocessing/upper_limb/loader.py  (preprocessing/upper_limb)
  MOVED    graph/upper_limb.py  →  preprocessing/upper_limb/graph.py  (preprocessing/upper_limb)
  MOVED    preprocessing/extract_face_dataset.py  →  preprocessing/face/extract.py  (preprocessing/face)
  MOVED    preprocessing/build_face_ctrgcn_dataset.py  →  preprocessing/face/build_tensors.py  (preprocessing/face)
  MOVED    preprocessing/split_face_dataset.py  →  preprocessing/face/split.py  (preprocessing/face)
  MOVED    dataset/face_loader.py  →  preprocessing/face/loader.py  (preprocessing/face)
  MOVED    graph/face_graph.py  →  preprocessing/face/graph.py  (preprocessing/face)
  MOVED    graph/face_landmark_mapping.py  →  preprocessing/face/landmark_mapping.py  (preprocessing/face)
  MOVED    preprocessing/extract_poststroke_dataset.py  →  preprocessing/post_stroke/extract.py  (preprocessing/post_stroke)
  MOVED    preprocessing/build_poststroke_ctrgcn_dataset.py  →  preprocessing/post_stroke/build_tensors.py  (preprocessing/post_stroke)
  MOVED    preprocessing/split_poststroke_dataset.py  →  preprocessing/post_stroke/split.py  (preprocessing/post_stroke)
  MOVED    preprocessing/organize_poststroke_dataset.py  →  preprocessing/post_stroke/organize.py  (preprocessing/post_stroke)
  MOVED    dataset/poststroke_loader.py  →  preprocessing/post_stroke/loader.py  (preprocessing/post_stroke)
  MOVED    graph/poststroke_graph.py  →  preprocessing/post_stroke/graph.py  (preprocessing/post_stroke)
  MOVED    preprocessing/tensor_statistics.py  →  preprocessing/shared/tensor_stats.py  (preprocessing/shared)
  MOVED    preprocessing/post_extraction_stats.py  →  preprocessing/shared/post_extraction_stats.py  (preprocessing/shared)
  REMOVED  dataset  (empty package after migration)
  REMOVED  graph  (empty package after migration)
  MOVED    training/train_lower_limb_ctrgcn.py  →  training/lower_limb/train.py  (training/lower_limb)
  MOVED    training/train_upper_limb_ctrgcn.py  →  training/upper_limb/train.py  (training/upper_limb)
  MOVED    training/train_face_ctrgcn.py  →  training/face/train.py  (training/face)
  MOVED    training/train_poststroke_ctrgcn.py  →  training/post_stroke/train.py  (training/post_stroke)
  MOVED    inference/predict_video.py  →  inference/lower_limb/predict.py  (inference/lower_limb)
  MOVED    inference/predict_upper_video.py  →  inference/upper_limb/predict.py  (inference/upper_limb)
  MOVED    inference/predict_face_video.py  →  inference/face/predict.py  (inference/face)
  MOVED    inference/predict_poststroke_video.py  →  inference/post_stroke/predict.py  (inference/post_stroke)
  MOVED    models/best_upper_limb_ctrgcn.pth  →  models/upper_limb/best_model.pth  (module checkpoint)
  MOVED    models/best_face_ctrgcn.pth  →  models/face/best_model.pth  (module checkpoint)
  MOVED    models/best_poststroke_ctrgcn.pth  →  models/post_stroke/best_model.pth  (module checkpoint)
  MOVED    results/lower_limb/v3  →  evaluation/lower_limb  (evaluation results)
  MOVED    results/upper_limb  →  evaluation/upper_limb  (evaluation results)
  MOVED    results/face  →  evaluation/face  (evaluation results)
  MOVED    results/post_stroke  →  evaluation/post_stroke  (evaluation results)
  REMOVED  results  (migrated to evaluation/)
  MOVED    automation/run_autonomous_pipeline.py  →  automation/collect_dataset.py  (renamed to semantic name)
  MOVED    tools/balance_and_train.py  →  automation/balance_and_train.py  (moved from tools/)
  REMOVED  tools  (empty after migration)

## Files Created
  CREATED  configs/upper_limb.yaml  (new config)
  CREATED  configs/face.yaml  (new config)
  CREATED  configs/post_stroke.yaml  (new config)
  CREATED  deployment/README.md  (deployment placeholder)

## Imports Updated
  IMPORTS  training/upper_limb/train.py
  IMPORTS  training/face/train.py
  IMPORTS  training/post_stroke/train.py
  IMPORTS  training/lower_limb/train.py
  IMPORTS  inference/upper_limb/predict.py
  IMPORTS  inference/face/predict.py
  IMPORTS  inference/post_stroke/predict.py
  IMPORTS  inference/lower_limb/predict.py
  IMPORTS  automation/run_face_pipeline.py
  IMPORTS  automation/run_upper_pipeline.py
  IMPORTS  automation/collect_dataset.py
  IMPORTS  automation/run_lower_pipeline.py

## Remaining Technical Debt
1. **Lower-limb dataset** — Still collecting tensors (balance_and_train.py running).
   Target: 40 videos/class. Current: ~109+ tensors across 9 classes.
2. **Upper-limb skeletons** — 34 tensors; no class-prefixed naming convention.
   Action: Apply same `{class}_{name}.npy` naming as lower-limb.
3. **Face dataset** — Only 15 tensors. Needs data collection to be useful.
4. **Post-stroke dataset** — 25 tensors, no train/test prefix naming.
5. **configs/** — `upper_limb.yaml`, `face.yaml`, `post_stroke.yaml` created as stubs.
   Training scripts do not yet read from YAML. Future: migrate hyperparams to YAML.
6. **deployment/** — Placeholder only. FastAPI + real-time inference not yet built.
7. **README.md** — Should be updated to reflect new structure (manual step).
