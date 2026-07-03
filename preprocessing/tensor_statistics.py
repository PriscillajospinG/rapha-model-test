"""
preprocessing/tensor_statistics.py
────────────────────────────────────
Validates all .npy tensors in processed_dataset_upper/skeletons/
and generates processed_dataset_upper/tensor_statistics.csv
"""
import csv
import sys
import numpy as np
from pathlib import Path
import pandas as pd

BASE_DIR      = Path(__file__).parent.parent
SKELETON_DIR  = BASE_DIR / "processed_dataset_upper" / "skeletons"
CSV_PATH      = BASE_DIR / "processed_dataset_upper" / "upper_limb_frame_labels.csv"
OUT_CSV       = BASE_DIR / "processed_dataset_upper" / "tensor_statistics.csv"

EXPECTED_SHAPE = (4, 300, 8, 1)

if not SKELETON_DIR.exists():
    print(f"[ERROR] Skeletons dir not found: {SKELETON_DIR}")
    sys.exit(1)

npy_files = sorted(SKELETON_DIR.glob("*.npy"))
print(f"Found {len(npy_files)} .npy files")

# Load frame CSV for frame counts and missing landmarks
df = pd.read_csv(CSV_PATH)
joint_x_cols = [c for c in df.columns if c.endswith("_x") and c.startswith("joint_")]

rows = []
bad  = []
for npy in npy_files:
    arr   = np.load(npy)
    shape = tuple(arr.shape)
    ok    = (shape == EXPECTED_SHAPE)
    if not ok:
        bad.append(f"{npy.name}  shape={shape}")

    sample_name = npy.stem
    # Match to CSV video
    matched = df[df["video_name"].str.startswith(sample_name)]
    frame_count       = len(matched)
    missing_landmarks = int((matched[joint_x_cols[0]] == -1).sum()) if frame_count > 0 else -1

    rows.append({
        "sample_name":       sample_name,
        "shape":             str(shape),
        "shape_ok":          ok,
        "frame_count":       frame_count,
        "missing_landmarks": missing_landmarks,
    })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"Tensor statistics saved to: {OUT_CSV}")

if bad:
    print(f"\n[WARN] {len(bad)} tensors with wrong shape:")
    for b in bad:
        print(f"  {b}")
else:
    print(f"\n✓  All {len(npy_files)} tensors have shape {EXPECTED_SHAPE}")
