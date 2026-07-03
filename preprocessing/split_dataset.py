"""
split_dataset.py
─────────────────────────────────────────────────────────────────────────────
Scans the processed_dataset/skeletons/ directory for *.npy skeleton files.
Derives labels from filename prefix (e.g., knee_ -> knee, hip_ -> hip).
Applies an 80/20 train/test split stratified by class, and generates:
    processed_dataset/train_labels.csv
    processed_dataset/test_labels.csv

Usage:
    python preprocessing/split_dataset.py
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup paths relative to project root
BASE_DIR = Path(__file__).parent.parent
SKELETON_DIR = BASE_DIR / "processed_dataset" / "skeletons"
OUTPUT_DIR = BASE_DIR / "processed_dataset"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Hardcoded class names mapping
CLASS_MAP = {
    "quadriceps": 0,
    "calf": 1,
    "leg_raise": 2,
    "toes": 3,
    "hip": 4,
    "hamstring": 5,
    "heel_slide": 6,
    "knee": 7,
    "ankle": 8
}

def split():
    if not SKELETON_DIR.exists():
        log.error("Skeletons directory not found at %s. Please run build_ctrgcn_dataset.py first.", SKELETON_DIR)
        sys.exit(1)

    npy_files = list(SKELETON_DIR.glob("*.npy"))
    log.info("Found %d skeleton (.npy) files.", len(npy_files))

    if len(npy_files) == 0:
        log.error("No skeleton files to split.")
        sys.exit(1)

    samples = []
    for file_path in npy_files:
        # Note: filenames may have trailing spaces, keep sample_name exactly as filename (minus extension)
        sample_name = file_path.stem
        
        # Determine class based on prefix
        assigned_label = None
        for class_name, label_id in CLASS_MAP.items():
            if sample_name.lower().startswith(f"{class_name}_"):
                assigned_label = label_id
                break
                
        if assigned_label is None:
            log.warning("Could not identify class from sample name prefix: %s. Defaulting to 0.", sample_name)
            assigned_label = 0
            
        samples.append({
            "sample_name": sample_name,
            "label": assigned_label
        })

    df = pd.DataFrame(samples)
    
    # Perform stratified split to keep class distribution similar
    try:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
    except ValueError:
        # Fallback if some classes have only 1 sample
        log.warning("Stratified split failed (likely due to classes with single samples). Falling back to random split.")
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )

    # Save to processed_dataset directory
    train_csv_path = OUTPUT_DIR / "train_labels.csv"
    test_csv_path = OUTPUT_DIR / "test_labels.csv"

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    log.info("Splits generated successfully:")
    log.info("  Train split: %d samples -> %s", len(train_df), train_csv_path)
    log.info("  Test split : %d samples -> %s", len(test_df), test_csv_path)

    # Print class distribution summaries
    log.info("Train class distribution:")
    for cls_name, cls_id in CLASS_MAP.items():
        count = len(train_df[train_df["label"] == cls_id])
        log.info("  %-15s (ID %d): %d", cls_name, cls_id, count)

if __name__ == "__main__":
    split()
