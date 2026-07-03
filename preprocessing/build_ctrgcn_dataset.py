"""
build_ctrgcn_dataset.py
─────────────────────────────────────────────────────────────────────────────
Reads the frame-level landmarks from processed_dataset/lower_limb_frame_labels.csv,
groups them by video, handles missing/failed frames, resamples/pads/loops to
exactly T=300 frames, extracts lower-limb joints (23-32), and constructs/saves
the CTR-GCN tensors of shape (4, 300, 10, 1) in processed_dataset/skeletons/.

Usage:
    python preprocessing/build_ctrgcn_dataset.py
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup paths relative to project root
BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "processed_dataset" / "lower_limb_frame_labels.csv"
OUTPUT_DIR = BASE_DIR / "processed_dataset" / "skeletons"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Constants
EXPECTED_C = 4   # x, y, z, visibility
EXPECTED_T = 300 # frames
EXPECTED_V = 10  # joints
EXPECTED_M = 1   # person

LOWER_LIMB_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def build_tensors():
    if not INPUT_CSV.exists():
        log.error("Input CSV not found at %s. Please run extraction first.", INPUT_CSV)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading frame labels from %s ...", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    
    # Group by video_name
    videos = df.groupby("video_name")
    log.info("Found %d video sequences in CSV.", len(videos))

    saved_count = 0

    for video_name, group in tqdm(videos, desc="Building CTR-GCN tensors"):
        # Sort group by frame number to ensure temporal order
        group = group.sort_values("frame")
        
        T_raw = len(group)
        if T_raw == 0:
            log.warning("Skipping empty video group: %s", video_name)
            continue
            
        # Extract the joints features: shape (T_raw, V, C)
        frames_data = np.zeros((T_raw, EXPECTED_V, EXPECTED_C), dtype=np.float32)
        
        for v_idx, jid in enumerate(LOWER_LIMB_JOINTS):
            # Columns: joint_{jid}_x, joint_{jid}_y, joint_{jid}_z, joint_{jid}_visibility
            cx = f"joint_{jid}_x"
            cy = f"joint_{jid}_y"
            cz = f"joint_{jid}_z"
            cv = f"joint_{jid}_visibility"
            
            # Map values, fill missing/failed sentinels (-1) with 0.0 for training stability
            frames_data[:, v_idx, 0] = group[cx].apply(lambda val: max(0.0, val) if val != -1 else 0.0).values
            frames_data[:, v_idx, 1] = group[cy].apply(lambda val: max(0.0, val) if val != -1 else 0.0).values
            frames_data[:, v_idx, 2] = group[cz].apply(lambda val: val if val != -1 else 0.0).values
            frames_data[:, v_idx, 3] = group[cv].apply(lambda val: max(0.0, val) if val != -1 else 0.0).values

        # Resample to T_out = 300
        if T_raw >= EXPECTED_T:
            # Uniform temporal sampling
            indices = np.linspace(0, T_raw - 1, EXPECTED_T, dtype=int)
            resampled = frames_data[indices]
        else:
            # Loop/tile/pad video to fill T_out
            repeats = -(-EXPECTED_T // T_raw) # ceiling division
            tiled = np.tile(frames_data, (repeats, 1, 1))
            resampled = tiled[:EXPECTED_T]

        # Reshape to (C, T, V, M)
        # (T, V, C) -> (C, T, V)
        tensor_gcn = np.transpose(resampled, (2, 0, 1))
        # Add M=1 dimension -> (C, T, V, M)
        tensor_gcn = np.expand_dims(tensor_gcn, axis=-1)

        # Save to skeletons dir
        # Ensure sample_name matches the format used by the dataset loader
        # Clean naming (remove trailing spaces if they got into video_name, but preserve the base structure)
        # Note: If the file name itself has a trailing space in video_name, it's safer to use the original
        # to match the CSV sample names. But let's check.
        sample_name = os.path.splitext(video_name)[0]
        
        npy_path = OUTPUT_DIR / f"{sample_name}.npy"
        np.save(npy_path, tensor_gcn)
        saved_count += 1

    log.info("Successfully converted and saved %d tensors of shape (%d, %d, %d, %d) to %s",
             saved_count, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M, OUTPUT_DIR)

if __name__ == "__main__":
    build_tensors()
