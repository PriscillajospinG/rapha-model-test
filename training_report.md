# Lower-Limb CTR-GCN — Final Training Report
Generated: 2026-07-15 14:40:54

## Dataset Summary
| Metric | Value |
|--------|-------|
| Total tensors | 345 |
| Train samples | 265 |
| Test samples  | 71 |
| Repaired tensors | 0 |
| Duplicates removed | 1 |

## Training
| Parameter | Value |
|-----------|-------|
| Epochs run | 25 (best) / 55 (total) |
| Best val accuracy | 15.49% |
| Training duration | 6.2 min |
| Device | mps |
| RAM | 17.2 GB |
| Platform | arm64 / Darwin |

## Evaluation
| Metric | Value |
|--------|-------|
| Accuracy | 15.49% |
| Macro F1 | 0.0577 |
| Weighted F1 | 0.0625 |
| Precision | 0.0383 |
| Recall | 0.1466 |
| Top-3 Accuracy | 35.21% |

## Per-class F1
| Class | F1 |
|-------|----|
| quadriceps   | 0.0000 |
| calf         | 0.2857 |
| leg_raise    | 0.0000 |
| toes         | 0.0000 |
| hip          | 0.0000 |
| hamstring    | 0.2333 |
| heel_slide   | 0.0000 |
| knee         | 0.0000 |
| ankle        | 0.0000 |

## Artefacts
- `models/lower_limb/best_model.pth`
- `models/lower_limb/last_model.pth`
- `models/lower_limb/best_model.onnx`
- `evaluation/lower_limb_final/confusion_matrix.png`
- `evaluation/lower_limb_final/accuracy_curve.png`
- `evaluation/lower_limb_final/loss_curve.png`
- `evaluation/lower_limb_final/per_class_f1.png`
- `evaluation/lower_limb_final/roc_curve.png`
- `evaluation/lower_limb_final/classification_report.txt`
- `evaluation/lower_limb_final/metrics.json`
- `evaluation/lower_limb_final/class_distribution.png`
- `evaluation/lower_limb_final/deployment_metadata.json`
