"""
preprocessing/build_face_ctrgcn_dataset.py
───────────────────────────────────────────────────────────────────────────
Reads the frame-level landmarks from:
    processed_dataset_face/face_frame_labels.csv

Groups them by video, replaces -1 sentinels with 0.0, resamples/pads
to exactly T=300 frames, and saves CTR-GCN tensors of shape:

    (3, 300, 33, 1)

to:
    processed_dataset_face/skeletons/

Tensor layout:
    C=3  : x, y, z  (no visibility — FaceMesh does not provide it)
    T=300: standardised frame count
    V=33 : facial landmark nodes
    M=1  : single tracked person

Usage:
    python preprocessing/build_face_ctrgcn_dataset.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
INPUT_CSV  = BASE_DIR / "processed_dataset_face" / "face_frame_labels.csv"
OUTPUT_DIR = BASE_DIR / "processed_dataset_face" / "skeletons"

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EXPECTED_C = 3    # x, y, z
EXPECTED_T = 300  # standardised frame count
EXPECTED_V = 33   # facial landmark nodes
EXPECTED_M = 1    # single person

FACE_NODES = list(range(EXPECTED_V))


def build_tensors() -> None:
    if not INPUT_CSV.exists():
        log.error(
            "Input CSV not found: %s\n"
            "Run preprocessing/extract_face_dataset.py first.",
            INPUT_CSV,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading frame labels from %s …", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    videos = df.groupby("video_name")
    log.info("Found %d video sequences in CSV.", len(videos))

    saved_count = 0
    shape_errors = 0

    for video_name, group in tqdm(videos, desc="Building face CTR-GCN tensors"):
        group = group.sort_values("frame")
        T_raw = len(group)

        if T_raw == 0:
            log.warning("Skipping empty group: %s", video_name)
            continue

        # Build raw array (T_raw, V, C)
        frames_data = np.zeros((T_raw, EXPECTED_V, EXPECTED_C), dtype=np.float32)

        for v_idx in FACE_NODES:
            cx = f"landmark_{v_idx}_x"
            cy = f"landmark_{v_idx}_y"
            cz = f"landmark_{v_idx}_z"

            # Replace -1 sentinel with 0.0
            def clean(val):
                v = float(val)
                return 0.0 if v == -1.0 else v

            frames_data[:, v_idx, 0] = group[cx].apply(clean).values
            frames_data[:, v_idx, 1] = group[cy].apply(clean).values
            frames_data[:, v_idx, 2] = group[cz].apply(clean).values

        # Resample to T_out = 300
        if T_raw >= EXPECTED_T:
            # Uniform subsampling
            indices   = np.linspace(0, T_raw - 1, EXPECTED_T, dtype=int)
            resampled = frames_data[indices]
        else:
            # Tile then truncate
            repeats   = -(-EXPECTED_T // T_raw)   # ceiling division
            tiled     = np.tile(frames_data, (repeats, 1, 1))
            resampled = tiled[:EXPECTED_T]

        # (T, V, C) → (C, T, V) → (C, T, V, M=1)
        tensor_gcn = np.transpose(resampled, (2, 0, 1))
        tensor_gcn = np.expand_dims(tensor_gcn, axis=-1)

        expected_shape = (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)
        if tensor_gcn.shape != expected_shape:
            log.error(
                "Shape mismatch for %s: got %s, expected %s",
                video_name, tensor_gcn.shape, expected_shape,
            )
            shape_errors += 1
            continue

        sample_name = os.path.splitext(video_name)[0]
        npy_path    = OUTPUT_DIR / f"{sample_name}.npy"
        np.save(npy_path, tensor_gcn)
        saved_count += 1

    log.info(
        "Saved %d face tensors of shape (%d, %d, %d, %d) to %s",
        saved_count,
        EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
        OUTPUT_DIR,
    )
    if shape_errors:
        log.warning("%d tensors had shape errors and were skipped.", shape_errors)


if __name__ == "__main__":
    build_tensors()
