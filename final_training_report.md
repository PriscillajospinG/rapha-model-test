# Lower-Limb CTR-GCN — Final Training Report
Generated: 2026-07-15 14:31:22

## Dataset Summary
| Class        | Videos | Tensors | Status |
|--------------|--------|---------|--------|
| ankle        |     38 |      43 | OK |
| calf         |     40 |      41 | OK |
| hamstring    |     35 |      39 | need +1 |
| heel_slide   |     36 |      37 | need +3 |
| hip          |     34 |      35 | need +5 |
| knee         |     31 |      39 | need +1 |
| leg_raise    |     34 |      31 | need +9 |
| quadriceps   |     29 |      32 | need +8 |
| toes         |     45 |      40 | OK |

Total tensors: 337  |  Train: 266  |  Test: 71

## Training Config
epochs=250, batch=32, lr=0.001, AdamW, CosineAnnealingLR,
label_smoothing=0.1, early_stop_patience=30,
WeightedRandomSampler, Weighted CrossEntropy

## Evaluation
*Training metrics not yet available.*

## Artefacts
- `models/best_lower_limb_final.pth`
- `results/final/confusion_matrix.png`
- `results/final/accuracy_curve.png`
- `results/final/loss_curve.png`
- `results/final/classification_report.txt`
- `results/final/metrics.json`
