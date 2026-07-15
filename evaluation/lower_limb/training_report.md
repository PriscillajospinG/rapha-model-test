# Rapha — Lower Limb CTR-GCN v2 Training Report

**Generated:** 2026-07-13 16:11:16

---

## Dataset Summary (Post-Increment)

| Metric | Value |
|---|---|
| Total Samples | 136 |
| Classes | 9 |
| Test Samples | 28 |

### Class Distribution

| Class | Samples |
|---|---|
| quadriceps           |      8 |
| calf                 |     23 |
| leg_raise            |      5 |
| toes                 |     23 |
| hip                  |     12 |
| hamstring            |      8 |
| heel_slide           |      7 |
| knee                 |     32 |
| ankle                |     18 |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch Size | 8 |
| Learning Rate | 0.001 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Augmentation | Gaussian noise + temporal flip + LR mirror |
| Device | MPS |
| Training Time | 236.7 s  (3.9 min) |

---

## V2 Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **28.57%** |
| Macro F1 | 0.0967 |
| Precision | 0.0741 |
| Recall | 0.1593 |
| Top-3 Accuracy | 64.29% |

---

## Model

| Property | Value |
|---|---|
| Architecture | CTR-GCN |
| Parameters | 2,159,394 |
| Size | 25.0 MB |
| Checkpoint | `models/best_lower_limb_ctrgcn_v2.pth` |

---

## Per-Class Performance

```
              precision    recall  f1-score   support

  quadriceps       0.00      0.00      0.00         2
        calf       0.43      0.60      0.50         5
   leg_raise       0.00      0.00      0.00         1
        toes       0.00      0.00      0.00         5
         hip       0.00      0.00      0.00         2
   hamstring       0.00      0.00      0.00         2
  heel_slide       0.00      0.00      0.00         1
        knee       0.24      0.83      0.37         6
       ankle       0.00      0.00      0.00         4

    accuracy                           0.29        28
   macro avg       0.07      0.16      0.10        28
weighted avg       0.13      0.29      0.17        28

```

---

*Rapha Physiotherapy AI*
