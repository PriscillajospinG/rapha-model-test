# CTR-GCN Model Comparison: V1 vs V2

**Generated:** 2026-07-13 15:58:05
**Evaluation device:** mps
**Test samples:** 28

---

## Summary

| Metric | V1 (original) | V2 (retrained) | Change |
|---|---|---|---|
| **Test Accuracy** | 28.57% | **28.57%** |   — |
| Macro F1 | 0.0967 | **0.0967** |   — |
| Precision | 0.0741 | 0.0741 |   — |
| Recall | 0.1593 | 0.1593 |   — |
| Top-3 Accuracy | 64.29% | 64.29% |   — |
| Parameters | 2,159,394 | 2,159,394 | — |
| Inference speed | 11.75 ms | 11.79 ms | slower ▼ 0.0454 (slower ▼4.54%) |

---

## Per-Class F1 Score Comparison

| Class | V1 F1 | V2 F1 | Change |
|---|---|---|---|
| quadriceps           | 0.0000 | 0.0000 | — 0.0000 |
| calf                 | 0.5000 | 0.5000 | — 0.0000 |
| leg_raise            | 0.0000 | 0.0000 | — 0.0000 |
| toes                 | 0.0000 | 0.0000 | — 0.0000 |
| hip                  | 0.0000 | 0.0000 | — 0.0000 |
| hamstring            | 0.0000 | 0.0000 | — 0.0000 |
| heel_slide           | 0.0000 | 0.0000 | — 0.0000 |
| knee                 | 0.3704 | 0.3704 | — 0.0000 |
| ankle                | 0.0000 | 0.0000 | — 0.0000 |

---

## Interpretation

### ✅ V2 outperforms V1

- Accuracy delta : +0.00%
- Macro F1 delta : +0.0000
- Speed delta    : +0.05 ms per sample

The new dataset has similar or slightly lower accuracy. Consider adding more balanced samples per class or running additional epochs.

---

## Checkpoints

| Version | Path |
|---|---|
| V1 | `models/best_lower_limb_ctrgcn.pth` |
| V2 | `models/best_lower_limb_ctrgcn_v2.pth` |

---

*Rapha Physiotherapy AI — Incremental Training Comparison*
