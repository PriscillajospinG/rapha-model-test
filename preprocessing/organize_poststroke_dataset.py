"""
preprocessing/organize_poststroke_dataset.py
───────────────────────────────────────────────────────────────────────────
Organizes post-stroke raw videos by copying them from the user's Downloads
folder into structured class subfolders inside the workspace directory:
    datasets/post_stroke/raw/

This classification is done dynamically based on filename patterns.

Usage:
    python preprocessing/organize_poststroke_dataset.py
"""

import csv
import logging
import re
import shutil
import sys
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent
RAW_SRC_DIR  = Path("/Users/priscillajosping/Downloads/Post Stroke Excercises")
OUTPUT_DIR   = BASE_DIR / "datasets/post_stroke/raw"
REPORT_CSV   = OUTPUT_DIR / "classification_report.csv"

# ─── Supported extensions ────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}

# ─── Target classes ─────────────────────────────────────────────────────────

ALL_CLASSES = [
    "reaching",
    "grasp_release",
    "shoulder_flexion",
    "shoulder_abduction",
    "elbow_flexion",
    "sit_to_stand",
    "weight_shift",
    "gait_training",
    "balance_training",
    "trunk_rotation",
    "unclassified"
]

# ─── Classification rules ────────────────────────────────────────────────────

RULES = [
    ("reaching", ["reaching"]),
    ("grasp_release", ["crumling", "grasp"]),
    ("shoulder_flexion", ["shoulder-flexio", "shoulder flexion", "shoulder_flexion"]),
    ("shoulder_abduction", ["horizontal range of motion", "horizontal-range-of-motion"]),
    ("elbow_flexion", ["elbow-flexion", "elbow flexion", "elbow_flexion", "elbow"]),
    ("sit_to_stand", ["sit-to-stand", "sit to stand", "getting up", "squat"]),
    ("weight_shift", ["weight-shift", "weight shift", "weight shifts", "weight-shifts"]),
    ("gait_training", ["heel", "lunge", "knee-flexion", "knee flexion", "calf", "crossing"]),
    ("balance_training", ["stance", "standing-feet", "eyes-open-closed", "eyes open closed"]),
    ("trunk_rotation", ["trunk", "side-to-side", "side to side"])
]

# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def classify_filename(filename: str) -> tuple[str, float, str]:
    """
    Classify a video file based on its name using pattern rules.
    Returns (class_name, confidence, match_reason).
    """
    name_lower = filename.lower()
    for class_name, keywords in RULES:
        for kw in keywords:
            if kw in name_lower:
                return class_name, 0.95, f"Filename contains '{kw}'"
    return "unclassified", 0.0, "No keyword rules matched"


def main() -> None:
    logger.info("=" * 70)
    logger.info("  Post-Stroke Rehabilitation Dataset Organizer")
    logger.info("=" * 70)

    if not RAW_SRC_DIR.exists():
        logger.error(f"Source directory not found: {RAW_SRC_DIR}")
        sys.exit(1)

    logger.info(f"Source directory: {RAW_SRC_DIR.resolve()}")
    logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Ensure output folders exist
    for cls in ALL_CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Collect source files
    src_files = [
        p for p in RAW_SRC_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    logger.info(f"Found {len(src_files)} video files in source directory.")

    records = []
    copied_count = 0

    for src_path in sorted(src_files):
        class_name, confidence, reason = classify_filename(src_path.name)
        dst_dir = OUTPUT_DIR / class_name
        dst_path = dst_dir / src_path.name

        logger.info(f"Copying {src_path.name} ➔ {class_name}/  ({reason})")
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {src_path.name}: {e}")

        records.append({
            "original_filename": src_path.name,
            "assigned_class": class_name,
            "confidence": confidence,
            "reason": reason
        })

    # Save classification report
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["original_filename", "assigned_class", "confidence", "reason"]
        )
        writer.writeheader()
        writer.writerows(records)

    logger.info("=" * 70)
    logger.info("  Organization Complete")
    logger.info(f"  Videos copied: {copied_count} / {len(src_files)}")
    logger.info(f"  Report CSV   : {REPORT_CSV.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
