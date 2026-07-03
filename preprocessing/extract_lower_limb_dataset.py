"""
extract_lower_limb_dataset.py
─────────────────────────────────────────────────────────────────────────────
Physiotherapy Lower-Limb CTR-GCN Dataset Labeling Pipeline

Recursively scans every video inside dataset_raw/, runs MediaPipe Pose on
every frame, extracts lower-limb joint landmarks (joints 23-32), and saves
a single frame-level labeled CSV to:

    processed_dataset/lower_limb_frame_labels.csv

Usage:
    python extract_lower_limb_dataset.py

Dependencies:
    pip install opencv-python mediapipe
"""

import csv
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "dataset_raw"
OUTPUT_DIR = BASE_DIR / "processed_dataset"
OUTPUT_CSV = OUTPUT_DIR / "lower_limb_frame_labels.csv"

# MediaPipe Pose landmark indices for lower limbs only
LOWER_LIMB_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# MediaPipe Pose settings
MP_STATIC_MODE        = False   # Video mode: uses tracking between frames (faster)
MP_MODEL_COMPLEXITY   = 1       # 0=Lite, 1=Full, 2=Heavy
MP_SMOOTH_LANDMARKS   = True
MP_ENABLE_SEGMENTATION = False
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF  = 0.5

# Missing landmark sentinel value
MISSING = -1

# ─── CSV Column Schema ─────────────────────────────────────────────────────────

BASE_COLS  = ["video_name", "frame", "label"]
JOINT_COLS = []
for _jid in LOWER_LIMB_JOINTS:
    JOINT_COLS += [
        f"joint_{_jid}_x",
        f"joint_{_jid}_y",
        f"joint_{_jid}_z",
        f"joint_{_jid}_visibility",
    ]
ALL_COLS = BASE_COLS + JOINT_COLS

# ─── Logger ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "extraction.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_videos(root: Path) -> list[tuple[Path, str]]:
    """
    Recursively collect all video files under `root`.
    Returns a sorted list of (video_path, label) tuples where label is the
    immediate parent folder name (i.e., the exercise class).
    Videos sitting directly inside `root` (no class sub-folder) are skipped.
    """
    results: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        # Skip videos directly under the root with no class folder
        if path.parent == root:
            logger.warning("Skipping video with no class sub-folder: %s", path)
            continue
        label = path.parent.name   # folder name → class label
        results.append((path, label))
    return results


def process_video(
    video_path: Path,
    label: str,
    pose,
    writer: csv.DictWriter,
) -> tuple[int, int]:
    """
    Process a single video frame-by-frame and write rows directly to the CSV.

    Rows are streamed to disk instead of buffered in memory — safe for
    thousands of frames.

    Returns:
        (frames_processed, failed_detections)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_name   = video_path.name

    logger.info(
        "  ▸ %-40s  frames=%-6d  fps=%.1f  label=%s",
        video_name, total_frames, fps, label,
    )

    frames_done = 0
    failed      = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB for MediaPipe (writeable flag avoids an internal copy)
        rgb_frame           = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results             = pose.process(rgb_frame)

        row: dict = {
            "video_name": video_name,
            "frame":      frames_done,
            "label":      label,
        }

        if results.pose_landmarks:
            lm_list = results.pose_landmarks.landmark
            for jid in LOWER_LIMB_JOINTS:
                lm                             = lm_list[jid]
                row[f"joint_{jid}_x"]          = round(lm.x, 6)
                row[f"joint_{jid}_y"]          = round(lm.y, 6)
                row[f"joint_{jid}_z"]          = round(lm.z, 6)
                row[f"joint_{jid}_visibility"] = round(lm.visibility, 6)
        else:
            failed += 1
            for jid in LOWER_LIMB_JOINTS:
                row[f"joint_{jid}_x"]          = MISSING
                row[f"joint_{jid}_y"]          = MISSING
                row[f"joint_{jid}_z"]          = MISSING
                row[f"joint_{jid}_visibility"] = MISSING

        writer.writerow(row)
        frames_done += 1

        # Inline progress every 100 frames
        if frames_done % 100 == 0:
            print(
                f"     [{video_name}]  frame {frames_done}/{total_frames}   \r",
                end="",
                flush=True,
            )

    cap.release()
    print()  # newline after \r progress line
    return frames_done, failed


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("  Physiotherapy Lower-Limb Dataset Extraction Pipeline")
    logger.info("=" * 70)

    # ── Validate input directory ─────────────────────────────────────────────
    if not RAW_DIR.exists():
        logger.error("Raw dataset directory not found: %s", RAW_DIR.resolve())
        sys.exit(1)

    # ── Collect all videos ───────────────────────────────────────────────────
    logger.info("[1/3] Scanning for videos in: %s", RAW_DIR.resolve())
    videos = collect_videos(RAW_DIR)

    if not videos:
        logger.error("No video files found under %s.", RAW_DIR)
        sys.exit(1)

    labels_found = sorted({lbl for _, lbl in videos})
    logger.info("      Found %d video(s)  |  classes: %s", len(videos), labels_found)

    # ── Prepare output ───────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Initialise MediaPipe Pose ────────────────────────────────────────────
    logger.info("[2/3] Initialising MediaPipe Pose …")
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(
        static_image_mode         = MP_STATIC_MODE,
        model_complexity          = MP_MODEL_COMPLEXITY,
        smooth_landmarks          = MP_SMOOTH_LANDMARKS,
        enable_segmentation       = MP_ENABLE_SEGMENTATION,
        min_detection_confidence  = MP_MIN_DETECTION_CONF,
        min_tracking_confidence   = MP_MIN_TRACKING_CONF,
    )

    # ── Stream frames directly to CSV ────────────────────────────────────────
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

            frames, failed = process_video(
                video_path = video_path,
                label      = label,
                pose       = pose_model,
                writer     = writer,
            )

            total_frames_all += frames
            total_failed_all += failed
            videos_done      += 1

            rate = failed / max(frames, 1) * 100
            logger.info(
                "  ✓ frames=%-6d  failed_detections=%-6d (%.1f%%)\n",
                frames, failed, rate,
            )

    pose_model.close()
    elapsed = time.perf_counter() - pipeline_start

    # ── Summary ──────────────────────────────────────────────────────────────
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
