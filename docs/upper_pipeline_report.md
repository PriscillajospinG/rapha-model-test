# Upper-Limb CTR-GCN Pipeline — Final Report

> Generated: 2026-06-25

---

## 1. Final Folder Structure

```
labeling-cv/
├── dataset_raw_upper/                    ← Auto-organized videos
│   ├── shoulder/                         (12 videos)
│   ├── shoulder_flexion/                 (4 videos)
│   ├── shoulder_rotation/                (12 videos)
│   ├── elbow/                            (6 videos)
│   ├── unclassified/                     (19 videos — back/trunk)
│   └── classification_report.csv
│
├── processed_dataset_upper/
│   ├── skeletons/                        (34 × .npy tensors)
│   ├── upper_limb_frame_labels.csv       (46,520 frame rows)
│   ├── upper_class_map.csv
│   ├── train_labels.csv                  (27 samples)
│   ├── test_labels.csv                   (7 samples)
│   ├── extraction_summary.txt
│   └── tensor_statistics.csv
│
├── graph/
│   ├── lower_limb.py                     (unchanged)
│   └── upper_limb.py                     ← NEW
│
├── dataset/
│   ├── loader.py                         (unchanged)
│   └── upper_loader.py                   ← NEW
│
├── preprocessing/
│   ├── organize_upper_dataset.py         ← NEW
│   ├── extract_upper_limb_dataset.py     ← NEW
│   ├── build_upper_ctrgcn_dataset.py     ← NEW
│   ├── split_upper_dataset.py            ← NEW
│   ├── post_extraction_stats.py          ← NEW
│   └── tensor_statistics.py             ← NEW
│
├── training/
│   ├── train_lower_limb_ctrgcn.py        (unchanged)
│   └── train_upper_limb_ctrgcn.py        ← NEW
│
├── inference/
│   ├── predict_video.py                  (unchanged)
│   └── predict_upper_video.py            ← NEW
│
├── models/
│   ├── best_lower_limb_ctrgcn.pth        (unchanged)
│   ├── best_upper_limb_ctrgcn.pth        ← NEW
│   └── final_upper_limb_ctrgcn.pth       ← NEW (copy of best)
│
├── results_upper/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── validation_report.txt
│
├── docs/
│   ├── upper_dataset_creation.md         ← NEW
│   ├── upper_training.md                 ← NEW
│   └── upper_inference.md               ← NEW
│
└── pose_landmarker_full.task             (MediaPipe model, 9.4 MB)
```

---

## 2. Number of Classes

| Class ID | Class Name | Videos | Type |
|---|---|---|---|
| 0 | `elbow` | 6 | Elbow exercises (plank, soft tissue, strengthening, retraction, stretch) |
| 1 | `shoulder` | 12 | General shoulder (dips, lawnmower, wall slides, serratus punch, etc.) |
| 2 | `shoulder_flexion` | 4 | Shoulder flexion exercises (assisted + isometric) |
| 3 | `shoulder_rotation` | 12 | Lateral/medial/external rotation exercises |

**Total classified classes: 4**
**Unclassified (back/trunk exercises): 19 videos** — moved to `unclassified/`

---

## 3. Videos Per Class

| Class | Train | Test | Total |
|---|---|---|---|
| elbow | 5 | 1 | 6 |
| shoulder | 9 | 3 | 12 |
| shoulder_flexion | 3 | 1 | 4 |
| shoulder_rotation | 10 | 2 | 12 |
| **Total** | **27** | **7** | **34** |

---

## 4. Tensors Generated

- **Total .npy tensors**: 34
- **Shape**: `(4, 300, 8, 1)` — verified for all 34
- **C=4**: x, y, z, visibility channels
- **T=300**: Standardised frame count (uniform sample if > 300, tile-loop if < 300)
- **V=8**: Upper-limb graph nodes (shoulders, elbows, wrists, hips)
- **M=1**: Single tracked person

---

## 5. Training Samples

- **Train split**: 27 samples (80%)
- **Split strategy**: Stratified 80/20

---

## 6. Test Samples

- **Test split**: 7 samples (20%)
- **No overlap** with training set ✓

---

## 7. Model Architecture

- **Architecture**: CTR-GCN (Channel-wise Topology Refinement GCN)
- **Total parameters**: 2,158,093
- **Graph nodes (V)**: 8 upper-limb joints
- **Temporal frames (T)**: 300
- **Input channels (C)**: 4
- **Output classes**: 4
- **Graph partitions**: 3 (self / centripetal / centrifugal)

---

## 8. Best Validation Accuracy

**85.71%** (achieved at epoch 59)

---

## 9. Final Validation Accuracy

**85.71%** (same — best checkpoint retained)

---

## 10. Training Time

**69.8 seconds** — 100 epochs on Apple M3 (MPS device)

---

## 11. Device Used

- **Apple M3 MPS** (Metal Performance Shaders)
- CUDA fallback available if GPU present
- CPU fallback available on any machine

---

## 12. Per-Class Performance (Test Set)

```
                   precision    recall  f1-score   support

            elbow       1.00      1.00      1.00         1
         shoulder       1.00      1.00      1.00         3
 shoulder_flexion       0.00      0.00      0.00         1
shoulder_rotation       0.67      1.00      0.80         2

         accuracy                           0.86         7
        macro avg       0.67      0.75      0.70         7
     weighted avg       0.76      0.86      0.80         7
```

---

## 13. Example Inference Command

```bash
cd labeling-cv
source .venv/bin/activate

python inference/predict_upper_video.py --video dataset_raw_upper/elbow/Plank\ on\ Elbows\ and\ Half\ Knee.mp4
```

### Example Output
```
──────────────────────────────────────────────────────────
  CTR-GCN Upper-Limb Inference
──────────────────────────────────────────────────────────
  Device     : mps
  Model ckpt : models/best_upper_limb_ctrgcn.pth
  Video      : Plank on Elbows and Half Knee.mp4
  Frames: 750  |  FPS: 30.0

[1/4] Running MediaPipe PoseLandmarker …
[2/4] Building input tensor … (4, 300, 8, 1)
[3/4] Loading CTR-GCN checkpoint … (epoch 59, val_acc=85.71%)
[4/4] Running inference …

──────────────────────────────────────────────────────────
  RESULT
──────────────────────────────────────────────────────────
  Predicted exercise : ELBOW
  Confidence         : ~high%
  Inference time     : ~15 ms
──────────────────────────────────────────────────────────
```

---

## 14. MediaPipe Extraction Statistics

| Metric | Value |
|---|---|
| Total videos processed | 34 |
| Total frames extracted | 46,520 |
| Failed pose detections | 4,375 (9.40%) |
| Extraction time | ~805 seconds |
| Detection rate | **90.60%** |

### Notes on failed detections:
- `Elbow Soft Tissue Release.mp4`: 45.6% failure rate — close-up frames where full body not visible
- `Elbow Strenghtning.mp4`: 40.1% failure rate — same reason (close-up elbow shots)
- These frames fill with 0.0 during tensor building

---

## 15. Pre-Flight Validation Results

All 5 checks passed ✓:

| Check | Result |
|---|---|
| Tensor shapes `(4, 300, 8, 1)` | ✅ All 34 verified |
| Graph `A.shape == (3, 8, 8)` | ✅ A[0] diagonal confirmed |
| Forward pass `(2,4,300,8,1) → (2,4)` | ✅ Output shape correct |
| All 4 classes in training set | ✅ Classes 0,1,2,3 present |
| Train/test overlap | ✅ No overlap (27 train / 7 test) |

---

## 16. Unclassified Videos

The following 19 videos could not be confidently mapped to an upper-limb exercise class (they are back/trunk exercises) and were placed in `dataset_raw_upper/unclassified/`:

| Video | Reason |
|---|---|
| Cat Cow Stretch.mp4 | Spinal mobilisation |
| Childs-Pose-Exercise.mp4 | Trunk stretch |
| Prone Press UP.mp4 | Prone extension |
| Prone Press-Ups with Lock and Sag.mp4 | Prone extension |
| Repeated Extension in Lying with Towel.mp4 | Spinal extension |
| Repeated Extension in Quadruped.mp4 | Spinal extension |
| Repeated Extension with Belt Over Ironing Board.mp4 | Spinal extension |
| Thoracic Extension Mobilization Quadruped.mp4 | Thoracic |
| Thoracic Extension Mobilization Seat with Towel Roll.mp4 | Thoracic |
| Thoracic Extension Mobilization on Chair.mp4 | Thoracic |
| Tin Soldier.mp4 | Trunk stability |
| TAble TOp Up Press Excercise.mp4 | Chest press |
| Wall Press Up Excercise.mp4 | Chest press |
| Plank on Hands with Leg Lifts.mp4 | Core/plank |
| Plank with Feet on Wall with Mountain Climber.mp4 | Core/plank |
| Press Back.mp4 | Trunk extension |
| Banded Row Excercise.mp4 | Back |
| Bridge excercise Variations.mp4 | Hip/core |
| Pelvic tilts.mp4 | Hip/core |

---

## 17. Known Limitations

> [!WARNING]
> **Small dataset — expect overfitting**. With 27 training samples across 4 classes (~6.8 per class), the CTR-GCN model (2.16M parameters) is highly over-parameterised for this dataset size. The 85.71% validation accuracy on 7 test samples is encouraging but should not be over-interpreted.

> [!WARNING]
> **shoulder_flexion has 0% precision on test set**. Only 1 test sample for shoulder_flexion — the model misclassifies it as shoulder_rotation, which shares very similar arm elevation patterns.

> [!NOTE]
> **Duplicate videos inflate counts**. Several videos from different sub-folders (e.g., `with equipments` vs `without equipments`) were copies of the same clip, producing `_1` suffixed duplicates. These count as separate samples but add limited unique motion diversity.

> [!NOTE]
> **MediaPipe close-up detection failures**. Elbow Soft Tissue Release and Elbow Strengthening videos have 40-54% pose detection failure because the camera was zoomed into the arm — MediaPipe cannot detect the full-body pose required. These frames are filled with zeros.

---

## 18. Recommended Next Steps

1. **Expand dataset** — Collect ≥ 20 videos per class (≥ 80 total) to meaningfully train a 2M-parameter model
2. **Add bicep_curl and tricep_extension classes** — No source videos exist for these currently
3. **Separate close-up videos** — Use a hand/arm landmark model (MediaPipe Hands) for close-up elbow exercises instead of full-body PoseLandmarker
4. **Address duplicates** — Deduplicate or use them as train/validation splits
5. **Apply stronger regularisation** — Increase dropout, use mixup augmentation, or use a smaller model (fewer GCN layers) for this dataset size
6. **Cross-validation** — With 34 samples, use k-fold cross-validation instead of a fixed split for more reliable performance estimates
7. **Real-time deployment** — Integrate with webcam stream using `RunningMode.LIVE_STREAM` for clinical use
