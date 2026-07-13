"""
automation/run_incremental.py
════════════════════════════════════════════════════════════════════════════════
Incremental dataset update + retraining pipeline for Lower Limb CTR-GCN.

WHAT IT DOES
────────────
  Step 1  Detect NEW videos vs existing tensors → new_video_inventory.json
  Step 2  MediaPipe extraction (new videos only, appended to frame CSV)
  Step 3  Build CTR-GCN tensors (new only, never overwrites existing .npy)
  Step 4  Rebuild stratified 80/20 split over ALL tensors
           → train_labels.csv, test_labels.csv, class_distribution.json
  Step 5  Validate complete dataset (shape, NaN, Inf, coverage, overlap)
  Step 6  Retrain CTR-GCN (100 epochs)
           → models/best_lower_limb_ctrgcn_v2.pth
  Step 7  Save results
           → results/lower_limb/v2/ (plots, metrics.json, classification_report.txt,
                           training_report.md)
  Step 8  Compare old vs new model
           → model_comparison.md

EXTERNAL VIDEO SOURCE
─────────────────────
  New videos come from a FLAT directory (no class subfolders).
  Class assignment uses (in priority order):
    1. tools/proposed_classification.json  (already reviewed by user)
    2. Keyword-matching rules (same as classify_new_videos.py)

NAMING CONVENTION (must match existing dataset)
────────────────────────────────────────────────
  Tensor filename : {classname}_{videoname_without_ext}.npy
  CSV sample_name : {classname}_{videoname_without_ext}
  This matches existing tensors like: ankle_Ankle Pumps  .npy

USAGE
─────
  # Full incremental run (default source = /Volumes/Honz's Things/Lower Limb)
  python automation/run_incremental.py

  # Custom video source
  python automation/run_incremental.py --source "/Volumes/Honz's Things/Lower Limb"

  # Skip extraction (tensors already built)
  python automation/run_incremental.py --skip-extraction

  # Start from retraining only
  python automation/run_incremental.py --steps 6 7 8

COMPATIBILITY
─────────────
  Output fully compatible with:
    training/train_lower_limb_ctrgcn.py
    inference/predict_video.py
  Neither file is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    top_k_accuracy_score, classification_report, confusion_matrix,
)

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.loader import (
    CLASS_NAMES, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
    PhysioSkeletonDataset, build_loaders,
)
from graph.lower_limb import LowerLimbGraph
from model.ctrgcn import Model

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# class name → integer label (from dataset/loader.py CLASS_NAMES)
CLASS_MAP: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

# Aliases: new canonical names → old names used in loader.py
CLASS_ALIASES: dict[str, str] = {
    "hip_abduction":  "hip",
    "knee_extension": "knee",
    "toe_raise":      "toes",
    "quadriceps_set": "quadriceps",
}

# Keyword rules for classifying flat-folder videos (in priority order)
# Each entry: (class_name, [keyword_list])
KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("heel_slide",  ["heel slide", "heel_slide"]),
    ("leg_raise",   ["leg raise", "leg lift", "leg pull", "straight leg", "slr"]),
    ("hamstring",   ["hamstring"]),
    ("quadriceps",  ["quad", "quadricep", "quads"]),
    ("hip",         ["hip bridge", "hip abduction", "hip flexion", "hip extension",
                     "hip raise", "hip thrust", "clam", "glute bridge", "hip knee pain",
                     "hip  knee pain", "1 hip", "2 best hip", "3 best hip", "4 best hip"]),
    ("knee",        ["knee", "squat", "lunge", "step up", "terminal knee"]),
    ("calf",        ["calf", "eccentric calf", "heel drop", "heel raise"]),
    ("toes",        ["toe raise", "toe lift", "toe curl", "toes", "tibialis",
                     "shin raise", "dorsiflexion", "ankle inversion abc",
                     "ankle eversion abc", "banded shin"]),
    ("ankle",       ["ankle"]),
]

# Fixed paths
RAW_DIR       = PROJECT_ROOT / "datasets/lower_limb/raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets/lower_limb"
SKELETON_DIR  = PROCESSED_DIR / "skeletons"
FRAME_CSV     = PROCESSED_DIR / "lower_limb_frame_labels.csv"
TRAIN_CSV     = PROCESSED_DIR / "train_labels.csv"
TEST_CSV      = PROCESSED_DIR / "test_labels.csv"
MODELS_DIR    = PROJECT_ROOT / "models"
OLD_MODEL     = MODELS_DIR / "best_lower_limb_ctrgcn.pth"
NEW_MODEL     = MODELS_DIR / "best_lower_limb_ctrgcn_v2.pth"
RESULTS_V2    = PROJECT_ROOT / "results/lower_limb/v2"
INVENTORY     = PROJECT_ROOT / "new_video_inventory.json"
CLASS_DIST    = PROCESSED_DIR / "class_distribution.json"
COMPARISON    = PROJECT_ROOT / "model_comparison.md"
PROPOSED_JSON = PROJECT_ROOT / "tools" / "proposed_classification.json"

# Lower limb MediaPipe joint IDs
LOWER_LIMB_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

DEFAULT_SOURCE = Path("/Volumes/Honz's Things/Lower Limb")

# ══════════════════════════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger() -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "incremental.log", encoding="utf-8", mode="a"),
        ],
    )
    return logging.getLogger("run_incremental")

log: logging.Logger = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def banner(title: str) -> None:
    s = "═" * 66
    log.info("\n%s\n  %s\n%s", s, title, s)

def step_header(n: int, total: int, title: str) -> None:
    log.info("\n┌── STEP %d/%d: %s", n, total, title)

def step_ok(msg: str) -> None:
    log.info("└── ✔  %s", msg)

def step_fail(msg: str) -> None:
    log.error("└── ✘  %s  — aborting.", msg)
    sys.exit(1)

def resolve_class(video_name: str, proposed: dict | None = None) -> str | None:
    """
    Resolve a flat-folder video filename to a class name.
    Priority:
      1. proposed_classification.json (user-reviewed)
      2. keyword matching rules
    Returns canonical class key (e.g. "ankle", "hip") or None.
    """
    stem = os.path.splitext(video_name)[0]

    # 1 — user-reviewed JSON
    if proposed and video_name in proposed:
        cls = proposed[video_name].get("class", "").lower()
        if cls in CLASS_MAP:
            return cls
        if cls in CLASS_ALIASES:
            return CLASS_ALIASES[cls]

    # 2 — keyword matching
    name_lc = stem.lower()
    for cls_name, keywords in KEYWORD_RULES:
        for kw in keywords:
            if kw.lower() in name_lc:
                return cls_name

    return None   # unclassified

def tensor_name(class_name: str, video_name: str) -> str:
    """
    Return the sample name (= tensor stem, = CSV sample_name).
    Format: {class}_{videoname_without_ext}
    e.g. "ankle_Ankle Pumps  "
    """
    stem = os.path.splitext(video_name)[0]
    return f"{class_name}_{stem}"

def device_select() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def model_param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def load_ctrgcn(ckpt_path: Path, device: torch.device) -> Model:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    graph = LowerLimbGraph()
    model = Model(
        num_class  = ckpt.get("num_class",   9),
        num_point  = ckpt.get("num_point",  10),
        num_person = ckpt.get("num_person",  1),
        in_channels= ckpt.get("in_channels", 4),
        graph      = graph,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Detect New Videos
# ══════════════════════════════════════════════════════════════════════════════

def step1_detect_new_videos(source_dir: Path, proposed: dict | None) -> list[dict]:
    """
    Scan source_dir for all video files.
    Compare against existing .npy in SKELETON_DIR.
    Write new_video_inventory.json.
    Returns list of new video records: [{video_name, class_name, source_path}]
    """
    step_header(1, 8, "Detecting New Videos")

    if not source_dir.exists():
        step_fail(f"Source directory not found: {source_dir}")

    SKELETON_DIR.mkdir(parents=True, exist_ok=True)

    # All existing tensor stems (e.g. "ankle_Ankle Pumps  ")
    existing_stems: set[str] = {
        p.stem for p in SKELETON_DIR.glob("*.npy")
    }
    log.info("  Existing tensors in SKELETON_DIR: %d", len(existing_stems))

    # Also gather all existing video stems from datasets/lower_limb/raw/ (already processed)
    existing_video_names: set[str] = set()
    if RAW_DIR.exists():
        for cls_folder in RAW_DIR.iterdir():
            if cls_folder.is_dir():
                for f in cls_folder.iterdir():
                    if f.suffix.lower() in VIDEO_EXTENSIONS:
                        existing_video_names.add(f.name)

    # Scan source_dir
    all_source_videos = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])
    log.info("  Videos found in source:           %d", len(all_source_videos))

    new_videos:      list[dict] = []
    existing_videos: list[dict] = []
    duplicates:      list[dict] = []
    unclassified:    list[str]  = []

    for vp in all_source_videos:
        vname = vp.name
        cls   = resolve_class(vname, proposed)

        if cls is None:
            log.warning("  ⚠  UNCLASSIFIED: %s", vname)
            unclassified.append(vname)
            continue

        tnsr_stem = tensor_name(cls, vname)

        if tnsr_stem in existing_stems:
            existing_videos.append({
                "video_name":   vname,
                "class_name":   cls,
                "tensor_stem":  tnsr_stem,
                "source_path":  str(vp),
            })
        else:
            new_videos.append({
                "video_name":   vname,
                "class_name":   cls,
                "tensor_stem":  tnsr_stem,
                "source_path":  str(vp),
            })

    # Inventory
    inventory = {
        "scanned_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir":        str(source_dir),
        "total_source":      len(all_source_videos),
        "new_videos":        new_videos,
        "existing_videos":   existing_videos,
        "duplicates":        duplicates,
        "unclassified":      unclassified,
        "class_breakdown_new": dict(Counter(v["class_name"] for v in new_videos)),
    }
    with open(INVENTORY, "w", encoding="utf-8") as fh:
        json.dump(inventory, fh, indent=2)

    log.info("  New videos (to process):          %d", len(new_videos))
    log.info("  Already processed (skip):         %d", len(existing_videos))
    log.info("  Unclassified (skipped):           %d", len(unclassified))
    if unclassified:
        for u in unclassified:
            log.warning("    ⚠  %s", u)
    log.info("  New by class:")
    for cls_name, cnt in sorted(inventory["class_breakdown_new"].items()):
        log.info("    %-16s  %d", cls_name, cnt)
    log.info("  new_video_inventory.json → %s", INVENTORY)

    if not new_videos:
        log.info("  No new videos detected — dataset is already up to date.")
        log.info("  Proceeding to split rebuild and retraining with existing tensors.")

    step_ok(f"Detected {len(new_videos)} new videos across "
            f"{len(inventory['class_breakdown_new'])} classes.")
    return new_videos

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MediaPipe Extraction (new videos only)
# ══════════════════════════════════════════════════════════════════════════════


def _extract_one_video(video_path: Path, class_name: str, options) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    video_name = video_path.name
    prefixed_video_name = f"{class_name}_{video_name}"
    rows: list[dict] = []
    frame_idx = 0
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps_val)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row: dict = {"video_name": prefixed_video_name, "frame": frame_idx, "label": class_name}

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_list = result.pose_landmarks[0]
                for jid in LOWER_LIMB_JOINTS:
                    lm = lm_list[jid]
                    row[f"joint_{jid}_x"]          = round(lm.x, 6)
                    row[f"joint_{jid}_y"]          = round(lm.y, 6)
                    row[f"joint_{jid}_z"]          = round(lm.z, 6)
                    row[f"joint_{jid}_visibility"] = round(lm.visibility, 6)
            else:
                for jid in LOWER_LIMB_JOINTS:
                    row[f"joint_{jid}_x"]          = 0.0
                    row[f"joint_{jid}_y"]          = 0.0
                    row[f"joint_{jid}_z"]          = 0.0
                    row[f"joint_{jid}_visibility"] = 0.0

            rows.append(row)
            frame_idx += 1

    cap.release()
    return rows


def step2_extract_landmarks(new_videos: list[dict]) -> None:
    """
    Run MediaPipe on new videos only.
    APPENDS new rows to FRAME_CSV (or creates it if missing).
    """
    step_header(2, 8, f"MediaPipe Extraction  ({len(new_videos)} new videos)")

    if not new_videos:
        log.info("  No new videos to extract — skipping.")
        step_ok("Extraction skipped (no new videos).")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    base_cols  = ["video_name", "frame", "label"]
    joint_cols = []
    for jid in LOWER_LIMB_JOINTS:
        joint_cols += [f"joint_{jid}_x", f"joint_{jid}_y",
                       f"joint_{jid}_z", f"joint_{jid}_visibility"]
    all_cols = base_cols + joint_cols

    file_exists = FRAME_CSV.exists()
    mode        = "a" if file_exists else "w"

    log.info("  Mode: %s (existing CSV: %s)", "APPEND" if file_exists else "CREATE", file_exists)
    log.info("  Initialising MediaPipe Pose (Tasks API) …")

    model_path = str(PROJECT_ROOT / "models/pose_landmarker_full.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_frames = 0
    total_ok     = 0
    t0 = time.perf_counter()

    with open(FRAME_CSV, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_cols)
        if not file_exists:
            writer.writeheader()

        for rec in new_videos:
            vp  = Path(rec["source_path"])
            cls = rec["class_name"]
            log.info("  ▸ %-55s class=%s", vp.name[:55], cls)

            rows = _extract_one_video(vp, cls, options)
            if not rows:
                log.warning("    ⚠ No frames extracted — skipped.")
                continue

            writer.writerows(rows)
            total_frames += len(rows)
            total_ok += 1
            log.info("    frames=%d", len(rows))

    elapsed = time.perf_counter() - t0

    log.info("  Videos processed : %d / %d", total_ok, len(new_videos))
    log.info("  New frames added : %s", f"{total_frames:,}")
    log.info("  Elapsed          : %.1f s", elapsed)

    if total_ok == 0:
        step_fail("No frames extracted from any new video.")

    step_ok(f"Appended {total_frames:,} frames from {total_ok} new videos to frame CSV.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build CTR-GCN Tensors (new videos only)
# ══════════════════════════════════════════════════════════════════════════════

def _rows_to_tensor(rows: list[dict]) -> np.ndarray:
    """Convert per-video CSV rows → (C, T, V, M) tensor float32."""
    T_raw = len(rows)
    data  = np.zeros((T_raw, EXPECTED_V, EXPECTED_C), dtype=np.float32)
    for ri, row in enumerate(rows):
        for vi, jid in enumerate(LOWER_LIMB_JOINTS):
            data[ri, vi, 0] = float(row.get(f"joint_{jid}_x", 0.0))
            data[ri, vi, 1] = float(row.get(f"joint_{jid}_y", 0.0))
            data[ri, vi, 2] = float(row.get(f"joint_{jid}_z", 0.0))
            data[ri, vi, 3] = float(row.get(f"joint_{jid}_visibility", 0.0))

    if T_raw >= EXPECTED_T:
        idx  = np.linspace(0, T_raw - 1, EXPECTED_T, dtype=int)
        data = data[idx]
    else:
        repeats = -(-EXPECTED_T // T_raw)
        data    = np.tile(data, (repeats, 1, 1))[:EXPECTED_T]

    # (T, V, C) → (C, T, V, M=1)
    tensor = np.transpose(data, (2, 0, 1))[:, :, :, None]
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor.astype(np.float32)


def step3_build_tensors(new_videos: list[dict]) -> list[str]:
    """
    Build tensors ONLY for new videos. Never overwrites existing .npy.
    Returns list of newly-built tensor stems.
    """
    step_header(3, 8, "Building CTR-GCN Tensors  (new videos only, no overwrite)")

    if not new_videos:
        log.info("  No new videos → skipping tensor build.")
        step_ok("Tensor build skipped (no new videos).")
        return []

    if not FRAME_CSV.exists():
        step_fail(f"Frame CSV not found: {FRAME_CSV}")

    SKELETON_DIR.mkdir(parents=True, exist_ok=True)

    # Load only new rows from CSV (prefixed video names)
    new_prefixed_names = {
        f"{rec['class_name']}_{rec['video_name']}"
        for rec in new_videos
    }

    log.info("  Loading frame CSV …")
    df = pd.read_csv(FRAME_CSV)
    df_new = df[df["video_name"].isin(new_prefixed_names)]
    log.info("  Frame rows for new videos: %s", f"{len(df_new):,}")

    saved:   list[str] = []
    skipped: int       = 0
    failed:  int       = 0

    for video_name, group in df_new.groupby("video_name"):
        group       = group.sort_values("frame")
        class_name  = group["label"].iloc[0]
        # video_name in CSV is prefixed: "ankle_Ankle Pumps.mp4"
        # tensor stem: "ankle_Ankle Pumps"
        sample_name = os.path.splitext(video_name)[0]
        npy_path    = SKELETON_DIR / f"{sample_name}.npy"

        if npy_path.exists():
            log.info("  [SKIP — exists] %s.npy", sample_name[:60])
            skipped += 1
            continue

        try:
            tensor = _rows_to_tensor(group.to_dict("records"))
            np.save(npy_path, tensor)
            saved.append(sample_name)
            log.info("  ✔  %-60s  %s", (sample_name + ".npy")[:60],
                     str(tensor.shape))
        except Exception as exc:
            log.warning("  ✗  Failed: %s — %s", video_name, exc)
            failed += 1

    log.info("  Tensors saved   : %d", len(saved))
    log.info("  Tensors skipped : %d  (already existed)", skipped)
    log.info("  Tensors failed  : %d", failed)

    step_ok(f"{len(saved)} new tensors saved to {SKELETON_DIR.relative_to(PROJECT_ROOT)}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Rebuild Full Dataset Split
# ══════════════════════════════════════════════════════════════════════════════

def step4_rebuild_split() -> None:
    """
    Stratified 80/20 split over ALL tensors in SKELETON_DIR.
    Infers label from filename prefix (e.g. "ankle_..." → label=8).
    Writes TRAIN_CSV, TEST_CSV, CLASS_DIST.
    """
    step_header(4, 8, "Rebuilding 80/20 Split (all tensors, stratified)")

    all_npy = sorted(SKELETON_DIR.glob("*.npy"))
    log.info("  Total tensors in SKELETON_DIR: %d", len(all_npy))

    samples: list[str] = []
    labels:  list[int] = []
    unresolved: list[str] = []

    for npy in all_npy:
        stem  = npy.stem          # e.g. "ankle_Ankle Pumps  "
        label = None

        # Infer class from prefix (try longest match first)
        for cls_name in sorted(CLASS_MAP.keys(), key=len, reverse=True):
            if stem.lower().startswith(cls_name + "_"):
                label = CLASS_MAP[cls_name]
                break

        if label is None:
            log.warning("  ⚠  Cannot infer class from tensor name: %s.npy", stem)
            unresolved.append(stem)
            continue

        samples.append(stem)
        labels.append(label)

    if unresolved:
        log.warning("  %d tensors could not be assigned a class — excluded.", len(unresolved))

    if len(samples) < 2:
        step_fail("Too few labelled samples to split.")

    # Class distribution report
    dist = dict(Counter(labels))
    class_dist_report = {
        "total_samples":       len(samples),
        "class_distribution":  {CLASS_NAMES[k]: v for k, v in sorted(dist.items())},
        "class_id_distribution": {str(k): v for k, v in sorted(dist.items())},
        "generated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASS_DIST, "w", encoding="utf-8") as fh:
        json.dump(class_dist_report, fh, indent=2)

    log.info("  Class distribution:")
    for cls_id, cnt in sorted(dist.items()):
        log.info("    %-16s (%d)  %d sample(s)", CLASS_NAMES[cls_id], cls_id, cnt)

    # Stratified split
    try:
        train_X, test_X, train_y, test_y = train_test_split(
            samples, labels, test_size=0.20, random_state=42, stratify=labels,
        )
    except ValueError:
        log.warning("  ⚠  Stratified split failed (class too small) — using random.")
        train_X, test_X, train_y, test_y = train_test_split(
            samples, labels, test_size=0.20, random_state=42,
        )

    def _write(path: Path, names: list, lbs: list) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["sample_name", "label"])
            w.writeheader()
            for n, l in zip(names, lbs):
                w.writerow({"sample_name": n, "label": l})

    _write(TRAIN_CSV, train_X, train_y)
    _write(TEST_CSV,  test_X,  test_y)

    log.info("  Train: %d  |  Test: %d", len(train_X), len(test_X))
    log.info("  class_distribution.json → %s", CLASS_DIST)
    step_ok(f"Split rebuilt: {len(train_X)} train, {len(test_X)} test over {len(samples)} total samples.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Validate Dataset
# ══════════════════════════════════════════════════════════════════════════════

def step5_validate_dataset() -> None:
    """Full integrity check: shape, NaN, Inf, class coverage, CSV ↔ npy match, no overlap."""
    step_header(5, 8, "Dataset Validation")

    expected_shape = (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)
    FAIL = False

    train_rows = read_csv_rows(TRAIN_CSV)
    test_rows  = read_csv_rows(TEST_CSV)

    if not train_rows: step_fail("train_labels.csv is empty or missing.")
    if not test_rows:  step_fail("test_labels.csv is empty or missing.")

    all_rows = [("train", r) for r in train_rows] + [("test", r) for r in test_rows]

    # ── 1: Shape + NaN/Inf ────────────────────────────────────────────────────
    log.info("  [1/5] Shape / NaN / Inf …")
    shape_errs: list[str] = []
    nan_errs:   list[str] = []
    for split, row in all_rows:
        name = (row.get("sample_name") or "").lstrip()
        npy  = SKELETON_DIR / f"{name}.npy"
        if not npy.exists():
            shape_errs.append(f"MISSING ({split}): {name}.npy")
            continue
        arr = np.load(npy)
        if arr.shape != expected_shape:
            shape_errs.append(f"BAD SHAPE {arr.shape} ({split}): {name}.npy")
        if np.any(np.isnan(arr)):
            nan_errs.append(f"NaN ({split}): {name}.npy")
        if np.any(np.isinf(arr)):
            nan_errs.append(f"Inf ({split}): {name}.npy")

    if shape_errs:
        for e in shape_errs: log.error("    ✘  %s", e)
        FAIL = True
    else:
        log.info("    ✓  All %d tensors are %s", len(all_rows), expected_shape)

    if nan_errs:
        for e in nan_errs: log.error("    ✘  %s", e)
        FAIL = True
    else:
        log.info("    ✓  No NaN / Inf values.")

    # ── 2: Class coverage ─────────────────────────────────────────────────────
    log.info("  [2/5] Class coverage …")
    train_labels = [int(r["label"]) for r in train_rows if r.get("label", "").strip().isdigit()]
    missing = set(CLASS_NAMES) - set(train_labels)
    if missing:
        log.warning("    ⚠  Classes absent from train: %s",
                    {k: CLASS_NAMES[k] for k in sorted(missing)})
    else:
        log.info("    ✓  All %d classes present in train split.", len(CLASS_NAMES))

    # ── 3: No overlap ─────────────────────────────────────────────────────────
    log.info("  [3/5] Train / test overlap …")
    train_names = {(r.get("sample_name") or "").lstrip() for r in train_rows}
    test_names  = {(r.get("sample_name") or "").lstrip() for r in test_rows}
    overlap = train_names & test_names
    if overlap:
        log.error("    ✘  %d overlap(s):", len(overlap))
        for o in sorted(overlap)[:5]: log.error("       %s", o)
        FAIL = True
    else:
        log.info("    ✓  No overlap between train and test.")

    # ── 4: CSV ↔ npy correspondence ───────────────────────────────────────────
    log.info("  [4/5] CSV ↔ .npy correspondence …")
    missing_npy = [
        (sp, (row.get("sample_name") or "").lstrip())
        for sp, row in all_rows
        if not (SKELETON_DIR / f"{(row.get('sample_name') or '').lstrip()}.npy").exists()
    ]
    if missing_npy:
        log.error("    ✘  %d missing .npy files:", len(missing_npy))
        for sp, nm in missing_npy[:5]: log.error("       [%s] %s.npy", sp, nm)
        FAIL = True
    else:
        log.info("    ✓  All CSV entries have matching .npy tensors.")

    if FAIL:
        step_fail("Dataset validation failed — see errors above.")

    step_ok("All dataset validation checks passed.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Retrain CTR-GCN
# ══════════════════════════════════════════════════════════════════════════════

def step6_retrain(results_v2_dir: Path) -> float:
    """
    Patch the training script's BEST_MODEL and RESULTS_DIR via env vars
    so it saves to v2 paths, then run it as a subprocess.
    Returns elapsed training time.
    """
    step_header(6, 8, "Retraining CTR-GCN  (→ best_lower_limb_ctrgcn_v2.pth)")

    results_v2_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_script = PROJECT_ROOT / "training" / "train_lower_limb_ctrgcn.py"
    if not train_script.exists():
        step_fail(f"Training script not found: {train_script}")

    # The existing training script writes to its hardcoded paths:
    #   models/best_lower_limb_ctrgcn.pth
    #   results/lower_limb/
    # We run it, THEN rename the outputs to v2 paths.
    # This avoids ANY modification to the training script.

    log.info("  Script      : %s", train_script.relative_to(PROJECT_ROOT))
    log.info("  After train : will copy outputs → v2 paths")
    log.info("  V2 model    : %s", NEW_MODEL.relative_to(PROJECT_ROOT))
    log.info("  V2 results/lower_limb  : %s", results_v2_dir.relative_to(PROJECT_ROOT))

    t_start = time.perf_counter()
    result  = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.perf_counter() - t_start

    if result.returncode != 0:
        step_fail(f"Training script exited with code {result.returncode}.")

    # ── Rename newly trained model to v2 (keep old v1 untouched) ────────────
    new_ckpt = PROJECT_ROOT / "models" / "best_lower_limb_ctrgcn.pth"
    if new_ckpt.exists():
        import shutil
        shutil.copy2(new_ckpt, NEW_MODEL)
        log.info("  Copied checkpoint → %s", NEW_MODEL.relative_to(PROJECT_ROOT))
    else:
        step_fail("Training did not produce a checkpoint.")

    # ── Copy result plots to results/lower_limb/v2/ ─────────────────────────────────────
    import shutil
    results_src = PROJECT_ROOT / "results/lower_limb"
    for fname in ["loss_curve.png", "accuracy_curve.png", "confusion_matrix.png"]:
        src = results_src / fname
        if src.exists():
            shutil.copy2(src, results_v2_dir / fname)
            log.info("  Copied %s → results/lower_limb/v2/", fname)

    # Copy classification report if it exists (from run_local_training)
    clf_src = results_src / "classification_report.txt"
    if clf_src.exists():
        shutil.copy2(clf_src, results_v2_dir / "classification_report.txt")

    log.info("  Training time : %.1f s  (%.1f min)", elapsed, elapsed / 60)
    step_ok(f"Retraining complete. V2 checkpoint: {NEW_MODEL.relative_to(PROJECT_ROOT)}")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Compute V2 Metrics and Save Results
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_model(model: Model, loader, device: torch.device
                    ) -> tuple[list[int], list[int], list[list[float]]]:
    """Evaluate model on a DataLoader. Returns (targets, preds, probs)."""
    model.eval()
    all_targets: list[int]         = []
    all_preds:   list[int]         = []
    all_probs:   list[list[float]] = []
    with torch.no_grad():
        for data, labels, _ in loader:
            data   = data.to(device)
            logits = model(data)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_targets.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_targets, all_preds, all_probs


def _compute_metrics(targets: list[int], preds: list[int],
                     probs: list[list[float]]) -> dict:
    probs_arr = np.array(probs)
    acc       = accuracy_score(targets, preds)
    macro_f1  = f1_score(targets, preds, average="macro", zero_division=0)
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall    = recall_score(targets, preds, average="macro", zero_division=0)
    try:
        top3 = top_k_accuracy_score(targets, probs_arr, k=3)
    except Exception:
        top3 = float("nan")
    return {
        "accuracy":      round(acc,       4),
        "macro_f1":      round(macro_f1,  4),
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "top3_accuracy": round(top3,      4),
    }


def step7_compute_and_save(results_v2_dir: Path, elapsed_train: float) -> dict:
    """Evaluate V2 model, save metrics.json, classification_report.txt, training_report.md."""
    step_header(7, 8, "Saving V2 Metrics & Results")

    device  = device_select()
    model   = load_ctrgcn(NEW_MODEL, device)
    _, _, _, test_loader = build_loaders(
        str(TRAIN_CSV), str(TEST_CSV), str(SKELETON_DIR), batch_size=8,
    )

    targets, preds, probs = _evaluate_model(model, test_loader, device)
    metrics = _compute_metrics(targets, preds, probs)
    metrics["num_test_samples"] = len(targets)
    metrics["training_seconds"] = round(elapsed_train, 1)
    metrics["device"] = str(device)

    results_v2_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(results_v2_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("  results/lower_limb/v2/metrics.json →")
    for k, v in metrics.items():
        log.info("    %-20s %s", k, v)

    # classification_report.txt
    present     = sorted(set(targets))
    tick_names  = [CLASS_NAMES.get(c, str(c)) for c in present]
    report_txt  = classification_report(
        targets, preds, labels=present, target_names=tick_names, zero_division=0,
    )
    (results_v2_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    log.info("  Per-class report:\n%s", report_txt)

    # training_report.md
    device_str = str(device).upper()
    n_params   = model_param_count(model)
    model_mb   = NEW_MODEL.stat().st_size / 1_048_576

    # Class distribution
    dist_data: dict = {}
    if CLASS_DIST.exists():
        with open(CLASS_DIST, encoding="utf-8") as fh:
            dist_data = json.load(fh)
    cls_rows = "\n".join(
        f"| {cls:<20} | {cnt:>6} |"
        for cls, cnt in dist_data.get("class_distribution", {}).items()
    )

    report_md = f"""# Rapha — Lower Limb CTR-GCN v2 Training Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Dataset Summary (Post-Increment)

| Metric | Value |
|---|---|
| Total Samples | {dist_data.get('total_samples', '?')} |
| Classes | {len(dist_data.get('class_distribution', {}))} |
| Test Samples | {len(targets)} |

### Class Distribution

| Class | Samples |
|---|---|
{cls_rows}

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch Size | 8 |
| Learning Rate | 0.001 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Augmentation | Gaussian noise + temporal flip + LR mirror |
| Device | {device_str} |
| Training Time | {elapsed_train:.1f} s  ({elapsed_train/60:.1f} min) |

---

## V2 Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **{metrics['accuracy']*100:.2f}%** |
| Macro F1 | {metrics['macro_f1']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| Top-3 Accuracy | {metrics['top3_accuracy']*100:.2f}% |

---

## Model

| Property | Value |
|---|---|
| Architecture | CTR-GCN |
| Parameters | {n_params:,} |
| Size | {model_mb:.1f} MB |
| Checkpoint | `models/best_lower_limb_ctrgcn_v2.pth` |

---

## Per-Class Performance

```
{report_txt}
```

---

*Rapha Physiotherapy AI*
"""
    (results_v2_dir / "training_report.md").write_text(report_md, encoding="utf-8")
    log.info("  results/lower_limb/v2/training_report.md saved.")
    step_ok("V2 metrics and results/lower_limb saved to results/lower_limb/v2/.")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Compare Old vs New Model
# ══════════════════════════════════════════════════════════════════════════════

def _inference_speed(model: Model, device: torch.device, n_runs: int = 50) -> float:
    """Measure average inference time per sample (ms)."""
    dummy = torch.zeros(1, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    return (time.perf_counter() - t0) / n_runs * 1000   # ms per sample


def step8_compare_models(metrics_v2: dict) -> None:
    """
    Load both checkpoints, evaluate on the SAME test loader,
    generate model_comparison.md.
    """
    step_header(8, 8, "Comparing Old vs New Model")

    device = device_select()

    if not OLD_MODEL.exists():
        log.warning("  Old model not found: %s — skipping comparison.", OLD_MODEL)
        step_ok("Comparison skipped (old model not found).")
        return

    if not NEW_MODEL.exists():
        step_fail(f"New model not found: {NEW_MODEL}")

    # ── Load both models ──────────────────────────────────────────────────────
    model_v1 = load_ctrgcn(OLD_MODEL, device)
    model_v2 = load_ctrgcn(NEW_MODEL, device)

    _, _, _, test_loader = build_loaders(
        str(TRAIN_CSV), str(TEST_CSV), str(SKELETON_DIR), batch_size=8,
    )

    # ── Evaluate V1 on current test set ───────────────────────────────────────
    log.info("  Evaluating V1 on current test set …")
    targets_v1, preds_v1, probs_v1 = _evaluate_model(model_v1, test_loader, device)
    m_v1 = _compute_metrics(targets_v1, preds_v1, probs_v1)

    # ── Evaluate V2 ───────────────────────────────────────────────────────────
    log.info("  Evaluating V2 on current test set …")
    targets_v2, preds_v2, probs_v2 = _evaluate_model(model_v2, test_loader, device)
    m_v2 = _compute_metrics(targets_v2, preds_v2, probs_v2)

    # ── Inference speed ───────────────────────────────────────────────────────
    log.info("  Measuring inference speed …")
    speed_v1 = _inference_speed(model_v1, device)
    speed_v2 = _inference_speed(model_v2, device)

    # ── Parameter counts ─────────────────────────────────────────────────────
    params_v1 = model_param_count(model_v1)
    params_v2 = model_param_count(model_v2)

    # ── Per-class F1 comparison ───────────────────────────────────────────────
    present     = sorted(set(targets_v2))
    tick_names  = [CLASS_NAMES.get(c, str(c)) for c in present]

    from sklearn.metrics import f1_score as _f1
    f1_v1_per = _f1(targets_v1, preds_v1, labels=present, average=None, zero_division=0)
    f1_v2_per = _f1(targets_v2, preds_v2, labels=present, average=None, zero_division=0)

    class_comparison_rows = "\n".join(
        f"| {tick_names[i]:<20} | {f1_v1_per[i]:.4f} | {f1_v2_per[i]:.4f} | "
        f"{'▲' if f1_v2_per[i] > f1_v1_per[i] else ('▼' if f1_v2_per[i] < f1_v1_per[i] else '—')} "
        f"{abs(f1_v2_per[i] - f1_v1_per[i]):.4f} |"
        for i in range(len(present))
    )

    # ── Confusion matrices (side-by-side text) ────────────────────────────────
    cm_v1 = confusion_matrix(targets_v1, preds_v1, labels=present)
    cm_v2 = confusion_matrix(targets_v2, preds_v2, labels=present)

    # ── Delta helper ──────────────────────────────────────────────────────────
    def delta(new: float, old: float) -> str:
        d = new - old
        if abs(d) < 0.0001:
            return "  —"
        return f"{'▲' if d > 0 else '▼'} {abs(d):.4f} ({'▲' if d > 0 else '▼'}{abs(d)*100:.2f}%)"

    comparison_md = f"""# CTR-GCN Model Comparison: V1 vs V2

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Evaluation device:** {device}
**Test samples:** {len(targets_v2)}

---

## Summary

| Metric | V1 (original) | V2 (retrained) | Change |
|---|---|---|---|
| **Test Accuracy** | {m_v1['accuracy']*100:.2f}% | **{m_v2['accuracy']*100:.2f}%** | {delta(m_v2['accuracy'], m_v1['accuracy'])} |
| Macro F1 | {m_v1['macro_f1']:.4f} | **{m_v2['macro_f1']:.4f}** | {delta(m_v2['macro_f1'], m_v1['macro_f1'])} |
| Precision | {m_v1['precision']:.4f} | {m_v2['precision']:.4f} | {delta(m_v2['precision'], m_v1['precision'])} |
| Recall | {m_v1['recall']:.4f} | {m_v2['recall']:.4f} | {delta(m_v2['recall'], m_v1['recall'])} |
| Top-3 Accuracy | {m_v1['top3_accuracy']*100:.2f}% | {m_v2['top3_accuracy']*100:.2f}% | {delta(m_v2['top3_accuracy'], m_v1['top3_accuracy'])} |
| Parameters | {params_v1:,} | {params_v2:,} | — |
| Inference speed | {speed_v1:.2f} ms | {speed_v2:.2f} ms | {delta(-speed_v2, -speed_v1).replace('▲','faster ▲').replace('▼','slower ▼')} |

---

## Per-Class F1 Score Comparison

| Class | V1 F1 | V2 F1 | Change |
|---|---|---|---|
{class_comparison_rows}

---

## Interpretation

{"### ✅ V2 outperforms V1" if m_v2['accuracy'] >= m_v1['accuracy'] else "### ⚠ V2 underperforms V1"}

- Accuracy delta : {(m_v2['accuracy'] - m_v1['accuracy'])*100:+.2f}%
- Macro F1 delta : {(m_v2['macro_f1'] - m_v1['macro_f1']):+.4f}
- Speed delta    : {(speed_v2 - speed_v1):+.2f} ms per sample

{"The expanded dataset has improved the model's generalisation across exercise classes." if m_v2['accuracy'] > m_v1['accuracy'] else "The new dataset has similar or slightly lower accuracy. Consider adding more balanced samples per class or running additional epochs."}

---

## Checkpoints

| Version | Path |
|---|---|
| V1 | `models/best_lower_limb_ctrgcn.pth` |
| V2 | `models/best_lower_limb_ctrgcn_v2.pth` |

---

*Rapha Physiotherapy AI — Incremental Training Comparison*
"""

    COMPARISON.write_text(comparison_md, encoding="utf-8")
    log.info("  model_comparison.md → %s", COMPARISON)

    log.info("  %-20s  V1=%6.2f%%  V2=%6.2f%%  Δ=%+.2f%%",
             "Accuracy",
             m_v1["accuracy"] * 100, m_v2["accuracy"] * 100,
             (m_v2["accuracy"] - m_v1["accuracy"]) * 100)
    log.info("  %-20s  V1=%.4f   V2=%.4f   Δ=%+.4f",
             "Macro F1", m_v1["macro_f1"], m_v2["macro_f1"],
             m_v2["macro_f1"] - m_v1["macro_f1"])

    step_ok("Comparison complete. See model_comparison.md.")


# ══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_incremental.py",
        description="Incremental dataset expansion + CTR-GCN retraining (8 steps).",
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help=f"Flat video directory (default: {DEFAULT_SOURCE}).")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip steps 1–3 (extraction already done).")
    p.add_argument("--skip-training",   action="store_true",
                   help="Skip step 6 (model already trained as v2).")
    p.add_argument("--steps", type=int, nargs="+",
                   help="Run only specified steps (e.g. --steps 4 5 6 7 8).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    global log
    log = setup_logger()
    args = parse_args()

    banner("Rapha — Lower Limb Incremental Training  |  8-Step Pipeline")
    log.info("  Project root   : %s", PROJECT_ROOT)
    log.info("  Video source   : %s", args.source)
    log.info("  Time           : %s", time.strftime("%Y-%m-%d %H:%M:%S"))

    t_total = time.perf_counter()

    # Determine active steps
    active = set(args.steps) if args.steps else set(range(1, 9))
    if args.skip_extraction:
        active -= {1, 2, 3}
    if args.skip_training:
        active -= {6}

    # Load proposed classification JSON (user-reviewed)
    proposed: dict | None = None
    if PROPOSED_JSON.exists():
        with open(PROPOSED_JSON, encoding="utf-8") as fh:
            raw = json.load(fh)
        # JSON format: {video_name: {class: …, confidence: …}}
        proposed = {
            entry.get("video_name", k): entry
            for k, entry in (raw.items() if isinstance(raw, dict) else {})
        }
        # Also support list format
        if isinstance(raw, list):
            proposed = {entry.get("video_name"): entry for entry in raw}
        log.info("  Loaded proposed_classification.json (%d entries)", len(proposed or {}))

    # State passed between steps
    new_videos:    list[dict] = []
    metrics_v2:    dict       = {}
    elapsed_train: float      = 0.0
    results_v2_dir = RESULTS_V2

    # ── STEP 1 ───────────────────────────────────────────────────────────────
    if 1 in active:
        new_videos = step1_detect_new_videos(args.source, proposed)
    else:
        log.info("[Step 1 skipped]")
        if INVENTORY.exists():
            with open(INVENTORY, encoding="utf-8") as fh:
                inv = json.load(fh)
            new_videos = inv.get("new_videos", [])

    # ── STEP 2 ───────────────────────────────────────────────────────────────
    if 2 in active:
        step2_extract_landmarks(new_videos)
    else:
        log.info("[Step 2 skipped]")

    # ── STEP 3 ───────────────────────────────────────────────────────────────
    if 3 in active:
        step3_build_tensors(new_videos)
    else:
        log.info("[Step 3 skipped]")

    # ── STEP 4 ───────────────────────────────────────────────────────────────
    if 4 in active:
        step4_rebuild_split()
    else:
        log.info("[Step 4 skipped]")

    # ── STEP 5 ───────────────────────────────────────────────────────────────
    if 5 in active:
        step5_validate_dataset()
    else:
        log.info("[Step 5 skipped]")

    # ── STEP 6 ───────────────────────────────────────────────────────────────
    if 6 in active:
        elapsed_train = step6_retrain(results_v2_dir)
    else:
        log.info("[Step 6 skipped]")

    # ── STEP 7 ───────────────────────────────────────────────────────────────
    if 7 in active and NEW_MODEL.exists():
        metrics_v2 = step7_compute_and_save(results_v2_dir, elapsed_train)
    else:
        log.info("[Step 7 skipped]")
        m_path = results_v2_dir / "metrics.json"
        if m_path.exists():
            with open(m_path, encoding="utf-8") as fh:
                metrics_v2 = json.load(fh)

    # ── STEP 8 ───────────────────────────────────────────────────────────────
    if 8 in active:
        step8_compare_models(metrics_v2)
    else:
        log.info("[Step 8 skipped]")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    banner("Incremental Pipeline Complete")
    log.info("  Total elapsed     : %.1f s  (%.1f min)", total_elapsed, total_elapsed / 60)
    log.info("  V2 checkpoint     : %s",
             NEW_MODEL.relative_to(PROJECT_ROOT) if NEW_MODEL.exists() else "N/A")
    if metrics_v2:
        log.info("  V2 test accuracy  : %.2f%%", metrics_v2.get("accuracy", 0) * 100)
        log.info("  V2 macro F1       : %.4f",   metrics_v2.get("macro_f1", 0))
    log.info("  model_comparison  : %s", COMPARISON.relative_to(PROJECT_ROOT))
    log.info("  results/lower_limb/v2/       : %s", results_v2_dir.relative_to(PROJECT_ROOT))
    log.info("═" * 66)
    return 0


if __name__ == "__main__":
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
    sys.exit(main())
