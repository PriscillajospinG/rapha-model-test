"""
preprocessing/split_face_dataset.py
───────────────────────────────────────────────────────────────────────────
Scans datasets/face/skeletons/ for *.npy skeleton files.
Derives labels from the face_frame_labels.csv.
Applies an 80/20 stratified train/test split and generates:

    datasets/face/train_labels.csv
    datasets/face/test_labels.csv
    datasets/face/face_class_map.csv

The class map is built dynamically from whatever classes appear in the
dataset, so it adapts automatically.

Usage:
    python preprocessing/split_face_dataset.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent
SKELETON_DIR = BASE_DIR / "datasets/face" / "skeletons"
OUTPUT_DIR   = BASE_DIR / "datasets/face"
INPUT_CSV    = OUTPUT_DIR / "face_frame_labels.csv"

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def split() -> None:
    # ── Guard: skeletons directory ────────────────────────────────────────────
    if not SKELETON_DIR.exists():
        log.error(
            "Skeletons directory not found: %s\n"
            "Run build_face_ctrgcn_dataset.py first.",
            SKELETON_DIR,
        )
        sys.exit(1)

    npy_files = list(SKELETON_DIR.glob("*.npy"))
    log.info("Found %d skeleton (.npy) files.", len(npy_files))

    if not npy_files:
        log.error("No skeleton files to split.")
        sys.exit(1)

    # ── Build class map from frame CSV ────────────────────────────────────────
    if not INPUT_CSV.exists():
        log.error("Frame CSV not found: %s", INPUT_CSV)
        sys.exit(1)

    frame_df    = pd.read_csv(INPUT_CSV, usecols=["video_name", "label"])
    video_label = (
        frame_df.drop_duplicates("video_name")
        .set_index("video_name")["label"]
        .to_dict()
    )  # {video_name (with ext) → class_string}

    # Dynamic class map — sorted for reproducibility
    all_classes = sorted(set(video_label.values()))
    class_map: dict[str, int] = {cls: idx for idx, cls in enumerate(all_classes)}
    log.info("Dynamic class map: %s", class_map)

    # ── Match .npy files to labels ────────────────────────────────────────────
    samples:   list[dict] = []
    unmatched: list[str]  = []

    for file_path in npy_files:
        sample_name = file_path.stem   # filename without .npy

        matched_label: str | None = None
        for vid_name, cls_str in video_label.items():
            if Path(vid_name).stem == sample_name:
                matched_label = cls_str
                break

        if matched_label is None:
            log.warning(
                "Could not match .npy '%s' to any video in CSV. Skipping.",
                sample_name,
            )
            unmatched.append(sample_name)
            continue

        samples.append({
            "sample_name": sample_name,
            "label":       class_map[matched_label],
            "class_name":  matched_label,
        })

    if not samples:
        log.error("No samples matched. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(samples)
    log.info("Matched %d samples for splitting.", len(df))
    if unmatched:
        log.warning("%d samples could not be matched: %s", len(unmatched), unmatched)

    # ── Stratified split ──────────────────────────────────────────────────────
    try:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
    except ValueError:
        log.warning(
            "Stratified split failed (likely single-sample classes). "
            "Falling back to random split."
        )
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    train_csv = OUTPUT_DIR / "train_labels.csv"
    test_csv  = OUTPUT_DIR / "test_labels.csv"

    train_df[["sample_name", "label"]].to_csv(train_csv, index=False)
    test_df[["sample_name",  "label"]].to_csv(test_csv,  index=False)

    # ── Save class map ────────────────────────────────────────────────────────
    class_map_path = OUTPUT_DIR / "face_class_map.csv"
    pd.DataFrame(
        [{"class_name": k, "label": v} for k, v in class_map.items()]
    ).to_csv(class_map_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("Splits generated:")
    log.info("  Train : %d samples → %s", len(train_df), train_csv)
    log.info("  Test  : %d samples → %s", len(test_df),  test_csv)
    log.info("  Class map saved : %s", class_map_path)

    log.info("Train class distribution:")
    for cls_name, cls_id in class_map.items():
        n = len(train_df[train_df["label"] == cls_id])
        log.info("  %-25s (ID %d): %d", cls_name, cls_id, n)

    log.info("Test class distribution:")
    for cls_name, cls_id in class_map.items():
        n = len(test_df[test_df["label"] == cls_id])
        log.info("  %-25s (ID %d): %d", cls_name, cls_id, n)


if __name__ == "__main__":
    split()
