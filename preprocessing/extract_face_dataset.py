"""
preprocessing/extract_face_dataset.py
───────────────────────────────────────────────────────────────────────────
Physiotherapy Facial Rehabilitation Dataset Labeling Pipeline

Recursively scans every video inside dataset_raw_face/, runs MediaPipe
FaceMesh on every frame, extracts 33 physiotherapy-specific facial
landmarks (x, y, z), and saves a single frame-level labeled CSV to:

    processed_dataset_face/face_frame_labels.csv

Landmark selection (33 nodes total):
    Left  Eyebrow  : MP 70, 63, 105, 66, 107  → nodes  0– 4
    Right Eyebrow  : MP 336,296, 334, 293, 300 → nodes  5– 9
    Eyes           : MP 33, 133, 362, 263, 159, 145, 386, 374 → nodes 10–17
    Cheeks         : MP 50, 280, 187, 411       → nodes 18–21
    Nose           : MP 1, 4, 168               → nodes 22–24
    Mouth          : MP 61, 291, 13, 14, 78, 308, 17, 0 → nodes 25–32

Feature channels: x, y, z  (NO visibility — FaceMesh does not provide it)
Failed detection: fill all coordinates with -1  (never skip frames)

Usage:
    python preprocessing/extract_face_dataset.py

Dependencies:
    pip install opencv-python mediapipe
"""

import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "dataset_raw_face"
OUTPUT_DIR = BASE_DIR / "processed_dataset_face"
OUTPUT_CSV = OUTPUT_DIR / "face_frame_labels.csv"

# ── Landmark configuration ────────────────────────────────────────────────────

sys.path.insert(0, str(BASE_DIR))
from graph.face_landmark_mapping import FACE_LANDMARK_IDS, NUM_FACE_NODES

# Folders to skip
SKIP_FOLDERS: set[str] = {"unclassified"}

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Sentinel for failed detection
MISSING = -1

# ── CSV column schema ─────────────────────────────────────────────────────────

BASE_COLS  = ["video_name", "frame", "label"]
JOINT_COLS = []
for _node in range(NUM_FACE_NODES):
    JOINT_COLS += [
        f"landmark_{_node}_x",
        f"landmark_{_node}_y",
        f"landmark_{_node}_z",
    ]
ALL_COLS = BASE_COLS + JOINT_COLS

# ── Logger ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "face_extraction.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_videos(root: Path) -> list[tuple[Path, str]]:
    """
    Recursively collect all video files under `root`.
    Returns sorted list of (video_path, label) tuples.
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


def process_video(
    video_path: Path,
    label: str,
    writer: csv.DictWriter,
    face_mesh,
) -> tuple[int, int]:
    """
    Process a single video frame-by-frame using MediaPipe FaceMesh.
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

    frames_done = 0
    failed      = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        row: dict = {
            "video_name": video_name,
            "frame":      frames_done,
            "label":      label,
        }

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_lms = results.multi_face_landmarks[0].landmark
            for node_idx, mp_id in enumerate(FACE_LANDMARK_IDS):
                lm = face_lms[mp_id]
                row[f"landmark_{node_idx}_x"] = round(lm.x, 6)
                row[f"landmark_{node_idx}_y"] = round(lm.y, 6)
                row[f"landmark_{node_idx}_z"] = round(lm.z, 6)
        else:
            failed += 1
            for node_idx in range(NUM_FACE_NODES):
                row[f"landmark_{node_idx}_x"] = MISSING
                row[f"landmark_{node_idx}_y"] = MISSING
                row[f"landmark_{node_idx}_z"] = MISSING

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


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("  Physiotherapy Face Dataset Extraction Pipeline")
    logger.info("=" * 70)

    if not RAW_DIR.exists():
        logger.error("Raw dataset directory not found: %s", RAW_DIR.resolve())
        logger.error(
            "Expected: dataset_raw_face/ (symlink or directory)\n"
            "Create with: ln -sf '/path/to/face raw' dataset_raw_face"
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

    # ── FaceMesh configuration ────────────────────────────────────────────────
    # Using mp.solutions.face_mesh (legacy API — stable in mediapipe >= 0.8)
    face_mesh_config = dict(
        static_image_mode        = False,   # video tracking mode
        max_num_faces            = 1,
        refine_landmarks         = True,    # enables iris + extra contour landmarks
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    logger.info("[2/3] Initialising MediaPipe FaceMesh …")
    logger.info("      Nodes  : %d", NUM_FACE_NODES)
    logger.info("      Channels: x, y, z (no visibility)")

    logger.info("[3/3] Processing videos → %s\n", OUTPUT_CSV.resolve())

    pipeline_start   = time.perf_counter()
    total_frames_all = 0
    total_failed_all = 0
    videos_done      = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ALL_COLS)
        writer.writeheader()

        # One FaceMesh context for all videos (re-initialised per video
        # because FaceMesh has internal state that can bleed across clips)
        for idx, (video_path, label) in enumerate(videos, start=1):
            logger.info("[Video %d/%d]  label='%s'", idx, len(videos), label)

            with mp.solutions.face_mesh.FaceMesh(**face_mesh_config) as face_mesh:
                frames, failed = process_video(
                    video_path = video_path,
                    label      = label,
                    writer     = writer,
                    face_mesh  = face_mesh,
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
        "  Failed face detections : %s  (%.2f%%)",
        f"{total_failed_all:,}",
        total_failed_all / max(total_frames_all, 1) * 100,
    )
    logger.info("  Elapsed time           : %.1fs", elapsed)
    logger.info("  Output CSV             : %s",   OUTPUT_CSV.resolve())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
