"""
preprocessing/extract_poststroke_dataset.py
───────────────────────────────────────────────────────────────────────────
Post-Stroke Rehabilitation CTR-GCN Dataset Labeling Pipeline

Recursively scans every video inside dataset_raw_poststroke/, runs MediaPipe
PoseLandmarker (using Tasks API — mediapipe >= 0.10) on every frame,
extracts post-stroke joint landmarks (joints 11-16, 23-28), and saves
a single frame-level labeled CSV to:

    processed_dataset_poststroke/poststroke_frame_labels.csv

Joint remapping (V=12):
    MP-11 Left Shoulder  → node 0
    MP-12 Right Shoulder → node 1
    MP-13 Left Elbow     → node 2
    MP-14 Right Elbow    → node 3
    MP-15 Left Wrist     → node 4
    MP-16 Right Wrist    → node 5
    MP-23 Left Hip       → node 6
    MP-24 Right Hip      → node 7
    MP-25 Left Knee      → node 8
    MP-26 Right Knee     → node 9
    MP-27 Left Ankle     → node 10
    MP-28 Right Ankle    → node 11

Usage:
    python preprocessing/extract_poststroke_dataset.py
"""

import csv
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "dataset_raw_poststroke"
OUTPUT_DIR = BASE_DIR / "processed_dataset_poststroke"
OUTPUT_CSV = OUTPUT_DIR / "poststroke_frame_labels.csv"

# Model path
MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"

# MediaPipe Pose landmark indices for post-stroke
POSTSTROKE_JOINTS_MP     = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # MP indices
POSTSTROKE_JOINTS_REMAP  = list(range(12))                                  # Graph nodes 0..11

# Folders to skip (not class directories)
SKIP_FOLDERS: set[str] = {"unclassified"}

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Missing landmark sentinel value
MISSING = -1

# ─── CSV Column Schema ────────────────────────────────────────────────────────

BASE_COLS  = ["video_name", "frame", "label"]
JOINT_COLS = []
for _node in POSTSTROKE_JOINTS_REMAP:
    JOINT_COLS += [
        f"joint_{_node}_x",
        f"joint_{_node}_y",
        f"joint_{_node}_z",
        f"joint_{_node}_visibility",
    ]
ALL_COLS = BASE_COLS + JOINT_COLS

# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "poststroke_extraction.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_videos(root: Path) -> list[tuple[Path, str]]:
    """
    Recursively collect all video files under `root`.
    Returns sorted list of (video_path, label) tuples.
    Videos in 'unclassified' or directly under root are skipped.
    """
    results: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if path.parent == root:
            logger.warning("Skipping video with no class sub-folder: %s", path)
            continue
        label = path.parent.name
        if label in SKIP_FOLDERS:
            logger.info("Skipping unclassified video: %s", path.name)
            continue
        results.append((path, label))
    return results


def process_video_tasks_api(
    video_path: Path,
    label: str,
    writer: csv.DictWriter,
    model_path: str,
) -> tuple[int, int]:
    """
    Process a single video frame-by-frame using the MediaPipe Tasks API.
    Returns (frames_processed, failed_detections).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_name   = video_path.name

    logger.info(
        "  ▸ %-45s  frames=%-6d  fps=%.1f  label=%s",
        video_name, total_frames, fps_val, label,
    )

    # Build PoseLandmarker in VIDEO mode (stateful tracking)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options   = base_options,
        running_mode   = RunningMode.VIDEO,
        num_poses      = 1,
        min_pose_detection_confidence = 0.5,
        min_pose_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
    )

    frames_done = 0
    failed      = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Tasks API: detect_for_video needs a timestamp in milliseconds
            timestamp_ms = int(frames_done * 1000 / fps_val)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row: dict = {
                "video_name": video_name,
                "frame":      frames_done,
                "label":      label,
            }

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_list = result.pose_landmarks[0]  # first (only) person
                for node_idx, mp_id in enumerate(POSTSTROKE_JOINTS_MP):
                    lm = lm_list[mp_id]
                    row[f"joint_{node_idx}_x"]          = round(lm.x, 6)
                    row[f"joint_{node_idx}_y"]          = round(lm.y, 6)
                    row[f"joint_{node_idx}_z"]          = round(lm.z, 6)
                    row[f"joint_{node_idx}_visibility"] = round(lm.visibility, 6)
            else:
                failed += 1
                for node_idx in POSTSTROKE_JOINTS_REMAP:
                    row[f"joint_{node_idx}_x"]          = MISSING
                    row[f"joint_{node_idx}_y"]          = MISSING
                    row[f"joint_{node_idx}_z"]          = MISSING
                    row[f"joint_{node_idx}_visibility"] = MISSING

            writer.writerow(row)
            frames_done += 1

            if frames_done % 100 == 0:
                print(
                    f"     [{video_name[:30]}]  frame {frames_done}/{total_frames}   \r",
                    end="",
                    flush=True,
                )

    cap.release()
    print()
    return frames_done, failed


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("  Physiotherapy Post-Stroke Dataset Extraction Pipeline")
    logger.info("=" * 70)

    if not RAW_DIR.exists():
        logger.error("Raw dataset directory not found: %s", RAW_DIR.resolve())
        sys.exit(1)

    if not MODEL_PATH.exists():
        logger.error(
            "PoseLandmarker model not found: %s\n"
            "Download it with:\n"
            "  curl -o pose_landmarker_full.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            MODEL_PATH,
        )
        sys.exit(1)

    logger.info("[1/3] Scanning for videos in: %s", RAW_DIR.resolve())
    videos = collect_videos(RAW_DIR)

    if not videos:
        logger.error("No video files found under %s.", RAW_DIR)
        sys.exit(1)

    labels_found = sorted({lbl for _, lbl in videos})
    logger.info("      Found %d video(s)  |  classes: %s", len(videos), labels_found)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[2/3] Using MediaPipe Tasks PoseLandmarker (VIDEO mode) …")
    logger.info("      Model: %s", MODEL_PATH)

    logger.info("[3/3] Processing videos → %s\n", OUTPUT_CSV.resolve())

    pipeline_start   = time.perf_counter()
    total_frames_all = 0
    total_failed_all = 0
    videos_done      = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ALL_COLS)
        writer.writeheader()

        for idx, (video_path, label) in enumerate(videos, start=1):
            logger.info("[Video %d/%d]  label='%s'", idx, len(videos), label)

            frames, failed = process_video_tasks_api(
                video_path = video_path,
                label      = label,
                writer     = writer,
                model_path = str(MODEL_PATH),
            )

            total_frames_all += frames
            total_failed_all += failed
            videos_done      += 1

            rate = failed / max(frames, 1) * 100
            logger.info(
                "  ✓ frames=%-6d  failed_detections=%-6d (%.1f%%)\n",
                frames, failed, rate,
            )

    elapsed = time.perf_counter() - pipeline_start

    logger.info("=" * 70)
    logger.info("  Pipeline Complete")
    logger.info("=" * 70)
    logger.info("  Videos processed       : %d",   videos_done)
    logger.info("  Total frames extracted : %s",   f"{total_frames_all:,}")
    logger.info(
        "  Failed pose detections : %s  (%.2f%%)",
        f"{total_failed_all:,}",
        total_failed_all / max(total_frames_all, 1) * 100,
    )
    logger.info("  Elapsed time           : %.1fs", elapsed)
    logger.info("  Output CSV             : %s",   OUTPUT_CSV.resolve())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
