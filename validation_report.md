# Lower-Limb Dataset Validation Report
Generated: from validation run

## Summary
| Metric | Value |
|--------|-------|
| Total tensors on disk | 152 |
| Valid tensors | 144 |
| Repaired tensors | 0 |
| Corrupted (removed) | 0 |
| Unclassified | 8 |
| Train samples | 113 |
| Test samples | 31 |
| Class imbalance ratio | 43.00x |
| Duplicate count | 0 |

## Tensors per Class
| Class        | Tensors | Share |
|--------------|---------|-------|
| ankle        |      43 |   29.9% |
| calf         |      15 |   10.4% |
| hamstring    |      16 |   11.1% |
| heel_slide   |      34 |   23.6% |
| hip          |      13 |    9.0% |
| knee         |      10 |    6.9% |
| leg_raise    |       4 |    2.8% |
| quadriceps   |       8 |    5.6% |
| toes         |       1 |    0.7% |

## Issues Found
- Corrupted: []
- Unclassified: 8 tensors (no class_ prefix match)

## Verdict
✅ Dataset valid — training can proceed.
