"""
preprocessing/organize_upper_dataset.py
───────────────────────────────────────────────────────────────────────────
Step 1 of the upper-limb pipeline.

Recursively scans the raw upper-limb video directory, classifies each video
by analysing its filename, and COPIES (not moves) it into the appropriate
exercise class subfolder under datasets/upper_limb/raw/.

Outputs
-------
dataset_raw_upper/
    shoulder/
    shoulder_flexion/
    shoulder_rotation/
    elbow/
    wrist/
    arm_raise/
    shoulder_abduction/
    bicep_curl/
    tricep_extension/
    unclassified/
dataset_raw_upper/classification_report.csv

Usage
-----
    python preprocessing/organize_upper_dataset.py
"""

import csv
import logging
import re
import shutil
import sys
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent
RAW_SRC_DIR  = BASE_DIR / "datasets/upper_limb/raw"   # place raw videos here (cross-platform)\n# NOTE: if your raw source lives elsewhere, override via the --src CLI flag or set\n#       the RAW_SRC_DIR environment variable: RAW_SRC_DIR=os.environ.get("RAW_SRC_DIR", ...)\n
OUTPUT_DIR   = BASE_DIR / "datasets/upper_limb/raw"
REPORT_CSV   = OUTPUT_DIR / "classification_report.csv"

# ─── Supported extensions ────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}

# ─── All target class folders (created even if empty for future data) ─────────

ALL_CLASSES = [
    "shoulder",
    "elbow",
    "wrist",
    "arm_raise",
    "shoulder_abduction",
    "shoulder_flexion",
    "shoulder_rotation",
    "bicep_curl",
    "tricep_extension",
    "unclassified",
]

# ─── Classification rules ────────────────────────────────────────────────────
#
# Order matters: more-specific patterns first.
# Each rule is (class_name, [keyword_patterns], confidence, reason_template)
#
# A keyword_pattern matches if ANY of its strings appears in the lowercased
# filename (after stripping the extension).
#
# The FIRST matching rule wins.

RULES: list[tuple[str, list[str], float, str]] = [
    # ── Wrist ──────────────────────────────────────────────────────────────
    (
        "wrist",
        ["wrist"],
        0.95,
        "Filename contains 'wrist'",
    ),
    # ── Bicep curl ─────────────────────────────────────────────────────────
    (
        "bicep_curl",
        ["bicep", "biceps", "curl"],
        0.95,
        "Filename contains bicep/curl keyword",
    ),
    # ── Tricep extension ───────────────────────────────────────────────────
    (
        "tricep_extension",
        ["tricep", "triceps"],
        0.95,
        "Filename contains 'tricep'",
    ),
    # ── Arm raise ──────────────────────────────────────────────────────────
    (
        "arm_raise",
        ["arm raise", "lateral raise", "front raise", "overhead raise"],
        0.90,
        "Filename contains arm-raise keyword",
    ),
    # ── Shoulder abduction ─────────────────────────────────────────────────
    (
        "shoulder_abduction",
        ["abduction", "abduct"],
        0.92,
        "Filename contains 'abduction'",
    ),
    # ── Shoulder flexion ───────────────────────────────────────────────────
    (
        "shoulder_flexion",
        [
            "shoulder flexion",
            "shoulder-flexion",
            "assisted flexion",
            "flexion in lying",
            "flexion with",
        ],
        0.92,
        "Filename indicates shoulder flexion exercise",
    ),
    # ── Shoulder rotation ──────────────────────────────────────────────────
    (
        "shoulder_rotation",
        [
            "lateral rotation",
            "medial rotation",
            "external rotation",
            "internal rotation",
            "rotation",
        ],
        0.90,
        "Filename indicates rotation exercise",
    ),
    # ── Elbow ──────────────────────────────────────────────────────────────
    (
        "elbow",
        ["elbow"],
        0.95,
        "Filename contains 'elbow'",
    ),
    # ── Shoulder (generic) ─────────────────────────────────────────────────
    (
        "shoulder",
        [
            "shoulder",
            "hand behind back",
            "lawnmower",
            "serratus",
            "wall perturbation",
            "scapula",
            "scapular",
            "wall slide",
            "push-up",
            "push up",
            "dips",
            "sleeper stretch",
            "supine",
            "side lying",
        ],
        0.80,
        "Filename indicates general shoulder/upper-limb exercise",
    ),
    # ── Back / trunk stabilisation (unclassified for upper-limb purposes) ──
    (
        "unclassified",
        [
            "cat cow",
            "child",
            "prone",
            "pelvic",
            "thoracic",
            "trunk",
            "bridge",
            "banded row",
            "table top",
            "wall press",
            "press back",
            "plank",
            "tin soldier",
            "break trunk",
            "repeated extension",
            "quadruped",
        ],
        0.65,
        "Filename suggests back/trunk exercise — unclassified for upper-limb",
    ),
]


# ─── Logger ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── Classification logic ────────────────────────────────────────────────────

def classify_video(stem: str) -> tuple[str, float, str]:
    """
    Classify a video by its filename stem (without extension).

    Returns (class_name, confidence, reason).
    Falls back to 'unclassified' with confidence 0.0.
    """
    text = stem.lower()

    for cls_name, keywords, confidence, reason in RULES:
        for kw in keywords:
            if kw in text:
                return cls_name, confidence, reason

    return "unclassified", 0.0, "No keyword matched — manual review required"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def collect_videos(root: Path) -> list[Path]:
    """Recursively collect all video files under root."""
    videos: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return videos


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 65)
    log.info("  Upper-Limb Dataset Organisation")
    log.info("=" * 65)

    # ── Validate source ───────────────────────────────────────────────────
    if not RAW_SRC_DIR.exists():
        log.error("Source directory not found: %s", RAW_SRC_DIR)
        sys.exit(1)

    # ── Create output folders ─────────────────────────────────────────────
    for cls in ALL_CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)
    log.info("Output root  : %s", OUTPUT_DIR.resolve())

    # ── Collect all videos ────────────────────────────────────────────────
    log.info("Scanning     : %s", RAW_SRC_DIR.resolve())
    videos = collect_videos(RAW_SRC_DIR)
    log.info("Videos found : %d", len(videos))

    if not videos:
        log.error("No video files found in source directory.")
        sys.exit(1)

    # ── Classify + copy ───────────────────────────────────────────────────
    report_rows: list[dict] = []
    class_counts: dict[str, int] = {cls: 0 for cls in ALL_CLASSES}

    for video_path in videos:
        stem      = video_path.stem
        cls, conf, reason = classify_video(stem)

        dest_dir  = OUTPUT_DIR / cls
        dest_path = dest_dir / video_path.name

        # Handle name collision
        if dest_path.exists():
            base   = dest_path.stem
            suffix = dest_path.suffix
            count  = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{base}_{count}{suffix}"
                count += 1

        shutil.copy2(video_path, dest_path)
        class_counts[cls] += 1

        report_rows.append({
            "video_name":      video_path.name,
            "source_path":     str(video_path.relative_to(RAW_SRC_DIR)),
            "predicted_class": cls,
            "confidence":      round(conf, 2),
            "reason":          reason,
        })

        log.info(
            "  %-50s  →  %-22s  (%.0f%%)",
            video_path.name[:50], cls, conf * 100,
        )

    # ── Write classification report ───────────────────────────────────────
    fieldnames = ["video_name", "source_path", "predicted_class", "confidence", "reason"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Organisation Complete")
    log.info("=" * 65)
    log.info("  Total videos processed : %d", len(videos))
    log.info("  Class distribution:")
    for cls in ALL_CLASSES:
        if class_counts[cls] > 0:
            log.info("    %-22s : %d", cls, class_counts[cls])
    log.info("  Report saved           : %s", REPORT_CSV.resolve())
    log.info("=" * 65)


if __name__ == "__main__":
    main()
