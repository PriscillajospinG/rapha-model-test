"""
preprocessing/build_upper_ctrgcn_dataset.py
───────────────────────────────────────────────────────────────────────────
Reads the frame-level landmarks from:
    datasets/upper_limb/upper_limb_frame_labels.csv

Groups them by video, handles missing frames, resamples/pads/loops to
exactly T=300 frames, and constructs/saves the CTR-GCN tensors of shape
(4, 300, 8, 1) in:
    datasets/upper_limb/skeletons/

Tensor layout:
    C=4  : x, y, z, visibility
    T=300: standardised frame count
    V=8  : upper-limb joints (nodes 0-7)
    M=1  : single tracked person

Usage:
    python preprocessing/build_upper_ctrgcn_dataset.py
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup paths relative to project root
BASE_DIR  = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "datasets/upper_limb" / "upper_limb_frame_labels.csv"
OUTPUT_DIR = BASE_DIR / "datasets/upper_limb" / "skeletons"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Constants
EXPECTED_C = 4    # x, y, z, visibility
EXPECTED_T = 300  # frames
EXPECTED_V = 8    # upper-limb joints
EXPECTED_M = 1    # person

UPPER_LIMB_NODES = list(range(EXPECTED_V))  # nodes 0..7


def build_tensors() -> None:
    if not INPUT_CSV.exists():
        log.error(
            "Input CSV not found at %s. Please run extract_upper_limb_dataset.py first.",
            INPUT_CSV,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading frame labels from %s …", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    videos = df.groupby("video_name")
    log.info("Found %d video sequences in CSV.", len(videos))

    saved_count = 0

    for video_name, group in tqdm(videos, desc="Building CTR-GCN tensors"):
        group = group.sort_values("frame")

        T_raw = len(group)
        if T_raw == 0:
            log.warning("Skipping empty video group: %s", video_name)
            continue

        # Extract joint features: shape (T_raw, V, C)
        frames_data = np.zeros((T_raw, EXPECTED_V, EXPECTED_C), dtype=np.float32)

        for v_idx in UPPER_LIMB_NODES:
            cx = f"joint_{v_idx}_x"
            cy = f"joint_{v_idx}_y"
            cz = f"joint_{v_idx}_z"
            cv = f"joint_{v_idx}_visibility"

            # Replace sentinel -1 with 0.0 for training stability
            frames_data[:, v_idx, 0] = group[cx].apply(
                lambda val: max(0.0, float(val)) if val != -1 else 0.0
            ).values
            frames_data[:, v_idx, 1] = group[cy].apply(
                lambda val: max(0.0, float(val)) if val != -1 else 0.0
            ).values
            frames_data[:, v_idx, 2] = group[cz].apply(
                lambda val: float(val) if val != -1 else 0.0
            ).values
            frames_data[:, v_idx, 3] = group[cv].apply(
                lambda val: max(0.0, float(val)) if val != -1 else 0.0
            ).values

        # Resample to T_out = 300
        if T_raw >= EXPECTED_T:
            indices  = np.linspace(0, T_raw - 1, EXPECTED_T, dtype=int)
            resampled = frames_data[indices]
        else:
            repeats   = -(-EXPECTED_T // T_raw)  # ceiling division
            tiled     = np.tile(frames_data, (repeats, 1, 1))
            resampled = tiled[:EXPECTED_T]

        # (T, V, C) → (C, T, V)
        tensor_gcn = np.transpose(resampled, (2, 0, 1))
        # Add M=1 → (C, T, V, M)
        tensor_gcn = np.expand_dims(tensor_gcn, axis=-1)

        assert tensor_gcn.shape == (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M), \
            f"Tensor shape mismatch for {video_name}: {tensor_gcn.shape}"

        sample_name = os.path.splitext(video_name)[0]
        npy_path    = OUTPUT_DIR / f"{sample_name}.npy"
        np.save(npy_path, tensor_gcn)
        saved_count += 1

    log.info(
        "Successfully saved %d tensors of shape (%d, %d, %d, %d) to %s",
        saved_count, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M, OUTPUT_DIR,
    )


if __name__ == "__main__":
    build_tensors()
