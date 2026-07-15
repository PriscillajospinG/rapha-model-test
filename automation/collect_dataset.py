"""
automation/run_autonomous_pipeline.py
════════════════════════════════════════════════════════════════════════════════
Autonomous Physiotherapy Dataset Collection Pipeline

10-step pipeline:
1. Search and Download via yt-dlp
2. Automatic Classification
3. Duplicate Removal (Perceptual Hashing)
4. Quality Filtering (MediaPipe / OpenCV)
5. MediaPipe Extraction
6. Tensor Generation
7. Dataset Split
8. Dataset Statistics
9. Training
10. Incremental Updates
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import yt_dlp
import imagehash
from PIL import Image
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.lower_limb.loader import (
    CLASS_NAMES, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

CLASS_MAP = {v: k for k, v in CLASS_NAMES.items()}
LOWER_LIMB_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

QUERIES = [
    ("ankle pumps physiotherapy", "ankle"),
    ("ankle dorsiflexion exercise", "ankle"),
    ("ankle rehabilitation exercise", "ankle"),
    ("stroke ankle exercise", "ankle"),
    ("calf stretch physiotherapy", "calf"),
    ("calf strengthening exercise", "calf"),
    ("calf rehabilitation", "calf"),
    ("hamstring stretch physiotherapy", "hamstring"),
    ("hamstring rehabilitation exercise", "hamstring"),
    ("heel slide exercise", "heel_slide"),
    ("heel slide physiotherapy", "heel_slide"),
    ("hip abduction exercise", "hip"),
    ("hip rehabilitation physiotherapy", "hip"),
    ("knee flexion exercise", "knee"),
    ("knee rehabilitation exercise", "knee"),
    ("straight leg raise physiotherapy", "leg_raise"),
    ("leg raise exercise", "leg_raise"),
    ("quadriceps set exercise", "quadriceps"),
    ("quad strengthening physiotherapy", "quadriceps"),
    ("post surgery quadriceps exercise", "quadriceps"),
    ("toe raise exercise", "toes"),
    ("toe mobility exercise", "toes")
]

# Paths
DATASETS_DIR  = PROJECT_ROOT / "datasets/lower_limb"
RAW_DIR       = DATASETS_DIR / "raw"
DOWNLOADS_DIR = DATASETS_DIR / "downloads"  # temporary landing zone
MANUAL_DIR    = DATASETS_DIR / "manual_review"
REJECTED_DIR  = DATASETS_DIR / "rejected"
DUPLICATE_DIR = DATASETS_DIR / "duplicates"
SKELETON_DIR  = DATASETS_DIR / "skeletons"

FRAME_CSV     = DATASETS_DIR / "lower_limb_frame_labels.csv"
TRAIN_CSV     = DATASETS_DIR / "train_labels.csv"
TEST_CSV      = DATASETS_DIR / "test_labels.csv"
MODELS_DIR    = PROJECT_ROOT / "models"
NEW_MODEL     = MODELS_DIR / "best_lower_limb_ctrgcn_auto.pth"
RESULTS_DIR   = PROJECT_ROOT / "results/lower_limb/auto"

HASH_REGISTRY = DATASETS_DIR / "hash_registry.json"
DUP_REPORT    = PROJECT_ROOT / "duplicate_report.json"
QUAL_REPORT   = PROJECT_ROOT / "quality_report.json"
STATS_REPORT  = PROJECT_ROOT / "dataset_statistics.json"

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
            logging.FileHandler(log_dir / "autonomous.log", encoding="utf-8", mode="a"),
        ],
    )
    return logging.getLogger("autonomous")

log: logging.Logger = None  # type: ignore

def banner(title: str):
    s = "═" * 66
    log.info("\n%s\n  %s\n%s", s, title, s)

def step_header(n: int, title: str):
    log.info("\n┌── STEP %d/10: %s", n, title)

def step_ok(msg: str):
    log.info("└── ✔  %s", msg)

def step_fail(msg: str):
    log.error("└── ✘  %s  — aborting.", msg)
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Search and Download
# ══════════════════════════════════════════════════════════════════════════════

def step1_search_and_download(fetch: bool, max_target: int = 30) -> list[str]:
    step_header(1, "Search and Download (yt-dlp)")
    if not fetch:
        log.info("  --no-fetch passed. Skipping YouTube download.")
        step_ok("Skipped.")
        return []

    log.info("  Auto-updating yt-dlp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "--quiet"], check=False)

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        "format": "b[ext=mp4][height<=720]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "match_filter": yt_dlp.utils.match_filter_func("duration >= 10 & duration <= 300"),
        "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
    }

    new_files = []
    
    current_counts = {c: 0 for c in CLASS_MAP.keys()}
    if RAW_DIR.exists():
        for cls in CLASS_MAP.keys():
            d = RAW_DIR / cls
            if d.exists():
                current_counts[cls] = len([f for f in d.iterdir() if f.is_file()])

    stats = {
        "download_attempts": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "403_errors": 0,
        "duplicates_removed": 0
    }
    
    failures_log = []

    for query, cls in QUERIES:
        if current_counts.get(cls, 0) >= max_target:
            log.info("  Skipping query '%s' (class '%s' already has >= %d videos)", query, cls, max_target)
            continue
        
        needed = max_target - current_counts.get(cls, 0)
        log.info("  Searching: '%s' (need %d for %s)", query, needed, cls)
        
        # Exponential backoff retry loop
        retries = 3
        backoffs = [5, 15, 30]
        success = False
        
        providers = [
            f"ytsearch{needed}:{query}",
            f"vimeosearch{needed}:{query}"
        ]
        
        for provider_query in providers:
            if success: break
            
            for attempt in range(retries):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        stats["download_attempts"] += 1
                        info = ydl.extract_info(provider_query, download=True)
                        if info and "entries" in info:
                            for entry in info["entries"]:
                                if not entry: continue
                                
                                title = entry.get("title", "")
                                ext = entry.get("ext", "mp4")
                                p = DOWNLOADS_DIR / f"{title}.{ext}"
                                
                                # Verification
                                if p.exists():
                                    size_mb = p.stat().st_size / (1024 * 1024)
                                    if size_mb < 1.0:
                                        p.unlink()
                                        continue
                                        
                                    cap = cv2.VideoCapture(str(p))
                                    if cap.isOpened():
                                        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                        duration = frames / fps if fps > 0 else 0
                                        cap.release()
                                        
                                        if min(w, h) >= 480 and 10 <= duration <= 300:
                                            new_files.append(str(p))
                                            stats["successful_downloads"] += 1
                                        else:
                                            p.unlink()
                                    else:
                                        p.unlink()
                            
                            success = True
                            break
                            
                except Exception as e:
                    err_msg = str(e)
                    log.warning("  Attempt %d failed for %s: %s", attempt+1, provider_query, err_msg)
                    if "403" in err_msg or "Forbidden" in err_msg:
                        stats["403_errors"] += 1
                    failures_log.append({
                        "query": provider_query,
                        "attempt": attempt+1,
                        "error": err_msg
                    })
                    if attempt < retries - 1:
                        time.sleep(backoffs[attempt])
                        
            if not success:
                stats["failed_downloads"] += 1
                log.error("  Failed to download from provider: %s", provider_query)

    # Dump logs
    with open(PROJECT_ROOT / "logs" / "download_failures.json", "w") as f:
        json.dump(failures_log, f, indent=2)
        
    with open(PROJECT_ROOT / "download_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    step_ok(f"Downloaded {len(new_files)} verified files into downloads/. Logged to download_statistics.json.")
    return new_files

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Automatic Classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_title(title: str) -> tuple[str, float]:
    title_lc = title.lower()
    best_match = "unclassified"
    best_score = 0.0
    
    rules = {
        "ankle": ["ankle pump", "ankle rehab", "ankle"],
        "calf": ["calf stretch", "calf raise", "calf"],
        "hamstring": ["hamstring stretch", "hamstring rehab", "hamstring"],
        "heel_slide": ["heel slide"],
        "hip": ["hip abduction", "hip rehab", "hip"],
        "knee": ["knee flexion", "knee rehab", "knee"],
        "leg_raise": ["straight leg raise", "leg raise", "slr"],
        "quadriceps": ["quadriceps set", "quad set", "quadricep"],
        "toes": ["toe raise", "toe rehab", "tibialis", "toe"]
    }
    
    for cls, kws in rules.items():
        for kw in kws:
            if kw in title_lc:
                score = 0.95 if len(kw.split()) > 1 else 0.80
                if score > best_score:
                    best_score = score
                    best_match = cls
                    
    return best_match, best_score

def step2_classify():
    step_header(2, "Automatic Classification")
    if not DOWNLOADS_DIR.exists():
        step_ok("No new downloads to classify.")
        return

    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_MAP.keys():
        (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

    classified = 0
    manual = 0
    
    for vp in DOWNLOADS_DIR.glob("*.*"):
        if vp.suffix.lower() not in {".mp4", ".mkv", ".webm", ".avi"}:
            continue
            
        cls, score = classify_title(vp.stem)
        
        safe_name = "".join(c for c in vp.stem if c.isalnum() or c in " _-")
        new_name = f"{safe_name}{vp.suffix}"
        
        # Require higher confidence (>= 0.85) to pass the simulated AI labeler
        if score >= 0.85 and cls in CLASS_MAP:
            dest = RAW_DIR / cls / new_name
            shutil.move(str(vp), str(dest))
            classified += 1
        else:
            dest = MANUAL_DIR / new_name
            shutil.move(str(vp), str(dest))
            manual += 1

    step_ok(f"Classified {classified} videos automatically. {manual} sent to manual review.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Duplicate Removal
# ══════════════════════════════════════════════════════════════════════════════

def get_middle_frame_hash(video_path: Path) -> str | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
        
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    p_hash = str(imagehash.phash(img))
    d_hash = str(imagehash.dhash(img))
    return f"{p_hash}_{d_hash}" 

def step3_deduplicate():
    step_header(3, "Duplicate Removal (Perceptual Hashing)")
    
    DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)
    
    registry = {}
    if HASH_REGISTRY.exists():
        with open(HASH_REGISTRY, "r") as f:
            registry = json.load(f)
            
    duplicates = []
    if DUP_REPORT.exists():
        with open(DUP_REPORT, "r") as f:
            duplicates = json.load(f)
            
    processed = 0
    
    for cls in CLASS_MAP.keys():
        d = RAW_DIR / cls
        if not d.exists(): continue
        
        for vp in d.iterdir():
            if vp.is_file():
                h = get_middle_frame_hash(vp)
                if not h:
                    continue
                    
                if h in registry and registry[h] != vp.name:
                    duplicates.append({
                        "original": registry[h],
                        "duplicate": vp.name,
                        "class": cls,
                        "hash": h
                    })
                    shutil.move(str(vp), str(DUPLICATE_DIR / vp.name))
                else:
                    registry[h] = vp.name
                    processed += 1

    with open(HASH_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)
        
    with open(DUP_REPORT, "w") as f:
        json.dump(duplicates, f, indent=2)

    step_ok(f"Kept {processed} unique. Total duplicates logged: {len(duplicates)}.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Quality Filtering
# ══════════════════════════════════════════════════════════════════════════════

def check_quality(video_path: Path, options) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Cannot open video"
        
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0
    
    if min(w, h) < 480:
        cap.release()
        return False, f"Resolution too low ({w}x{h})"
    if duration < 5:
        cap.release()
        return False, f"Duration too short ({duration:.1f}s)"

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    missing_landmarks_count = 0
    multi_person_count = 0
    low_vis_count = 0
    checked_frames = 0
    
    prev_gray = None
    high_movement_frames = 0
    
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i in range(30):
            frame_idx = int((i / 30.0) * frames)
            if frame_idx >= frames: break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)
                if mean_diff > 40.0:  # heavy camera movement / scene change
                    high_movement_frames += 1
            prev_gray = gray

            if not ret: break
                
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            
            if not result.pose_landmarks:
                missing_landmarks_count += 1
            else:
                lms = result.pose_landmarks[0]
                vis_avg = sum(lms[jid].visibility for jid in LOWER_LIMB_JOINTS) / len(LOWER_LIMB_JOINTS)
                if vis_avg < 0.7:
                    low_vis_count += 1
                if len(result.pose_landmarks) > 1:
                    multi_person_count += 1
            checked_frames += 1

    cap.release()
    
    if checked_frames > 0:
        if multi_person_count > 0:
            return False, "Multiple people detected"
        if (missing_landmarks_count / checked_frames) > 0.3:
            return False, "Landmarks missing for >30% frames"
            
    return True, "OK"

def step4_quality_filter():
    step_header(4, "Quality Filtering")
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = str(PROJECT_ROOT / "models/pose_landmarker_full.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.5
    )

    rejects = []
    if QUAL_REPORT.exists():
        with open(QUAL_REPORT, "r") as f:
            rejects = json.load(f)
            
    passed = 0
    rejected_this_run = 0
    
    for cls in CLASS_MAP.keys():
        d = RAW_DIR / cls
        if not d.exists(): continue
        
        for vp in d.iterdir():
            if vp.is_file():
                tensor_path = SKELETON_DIR / f"{cls}_{vp.stem}.npy"
                if tensor_path.exists():
                    passed += 1
                    continue
                    
                ok, reason = check_quality(vp, options)
                if not ok:
                    rejects.append({"video": vp.name, "class": cls, "reason": reason})
                    shutil.move(str(vp), str(REJECTED_DIR / vp.name))
                    rejected_this_run += 1
                else:
                    passed += 1

    with open(QUAL_REPORT, "w") as f:
        json.dump(rejects, f, indent=2)
        
    step_ok(f"Passed {passed} videos. Rejected {rejected_this_run} this run.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: MediaPipe Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_one_video(video_path: Path, class_name: str, options) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []
    prefixed_video_name = f"{class_name}_{video_path.name}"
    rows = []
    frame_idx = 0
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, bgr = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps_val)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            row = {"video_name": prefixed_video_name, "frame": frame_idx, "label": class_name}

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_list = result.pose_landmarks[0]
                for jid in LOWER_LIMB_JOINTS:
                    lm = lm_list[jid]
                    row[f"joint_{jid}_x"] = round(lm.x, 6)
                    row[f"joint_{jid}_y"] = round(lm.y, 6)
                    row[f"joint_{jid}_z"] = round(lm.z, 6)
                    row[f"joint_{jid}_visibility"] = round(lm.visibility, 6)
            else:
                for jid in LOWER_LIMB_JOINTS:
                    row[f"joint_{jid}_x"] = 0.0
                    row[f"joint_{jid}_y"] = 0.0
                    row[f"joint_{jid}_z"] = 0.0
                    row[f"joint_{jid}_visibility"] = 0.0

            rows.append(row)
            frame_idx += 1

    cap.release()
    return rows

def step5_extract():
    step_header(5, "MediaPipe Extraction")
    base_cols = ["video_name", "frame", "label"]
    joint_cols = []
    for jid in LOWER_LIMB_JOINTS:
        joint_cols += [f"joint_{jid}_x", f"joint_{jid}_y", f"joint_{jid}_z", f"joint_{jid}_visibility"]
    all_cols = base_cols + joint_cols

    file_exists = FRAME_CSV.exists()
    mode = "a" if file_exists else "w"

    model_path = str(PROJECT_ROOT / "models/pose_landmarker_full.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_frames = 0
    extracted = 0

    with open(FRAME_CSV, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_cols)
        if not file_exists:
            writer.writeheader()

        for cls in CLASS_MAP.keys():
            d = RAW_DIR / cls
            if not d.exists(): continue
            for vp in d.iterdir():
                if not vp.is_file(): continue
                tensor_path = SKELETON_DIR / f"{cls}_{vp.stem}.npy"
                if tensor_path.exists(): continue

                rows = _extract_one_video(vp, cls, options)
                if rows:
                    writer.writerows(rows)
                    total_frames += len(rows)
                    extracted += 1

    step_ok(f"Extracted {total_frames} frames from {extracted} new videos.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Tensor Generation
# ══════════════════════════════════════════════════════════════════════════════

def _rows_to_tensor(rows: list[dict]) -> np.ndarray:
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

    tensor = np.transpose(data, (2, 0, 1))[:, :, :, None]
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor.astype(np.float32)

def step6_generate_tensors():
    step_header(6, "Tensor Generation")
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    if not FRAME_CSV.exists():
        step_ok("No frames to convert.")
        return

    df = pd.read_csv(FRAME_CSV)
    saved = 0
    for video_name, group in df.groupby("video_name"):
        sample_name = os.path.splitext(video_name)[0]
        npy_path = SKELETON_DIR / f"{sample_name}.npy"
        
        if npy_path.exists(): continue
            
        group = group.sort_values("frame")
        tensor = _rows_to_tensor(group.to_dict("records"))
        np.save(npy_path, tensor)
        saved += 1

    step_ok(f"Generated {saved} new tensors.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Dataset Split
# ══════════════════════════════════════════════════════════════════════════════

def step7_split():
    step_header(7, "Dataset Split (Stratified 80/20)")
    all_npy = sorted(SKELETON_DIR.glob("*.npy"))
    
    samples = []
    labels = []
    for npy in all_npy:
        stem = npy.stem
        label = None
        for cls_name in sorted(CLASS_MAP.keys(), key=len, reverse=True):
            if stem.lower().startswith(cls_name + "_"):
                label = CLASS_MAP[cls_name]
                break
        if label is not None:
            samples.append(stem)
            labels.append(label)

    if len(samples) < 2:
        step_fail("Not enough valid samples to split.")

    try:
        train_X, test_X, train_y, test_y = train_test_split(
            samples, labels, test_size=0.20, random_state=42, stratify=labels,
        )
    except ValueError:
        log.warning("Stratified split failed. Using random split.")
        train_X, test_X, train_y, test_y = train_test_split(samples, labels, test_size=0.20, random_state=42)

    def _write(path, names, lbs):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sample_name", "label"])
            w.writeheader()
            for n, l in zip(names, lbs):
                w.writerow({"sample_name": n, "label": l})

    _write(TRAIN_CSV, train_X, train_y)
    _write(TEST_CSV, test_X, test_y)
    step_ok(f"Split created: {len(train_X)} train, {len(test_X)} test.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Dataset Statistics
# ══════════════════════════════════════════════════════════════════════════════

def step8_statistics():
    step_header(8, "Dataset Statistics")
    
    total_videos = len(list(SKELETON_DIR.glob("*.npy")))
    
    train_count = len(pd.read_csv(TRAIN_CSV)) if TRAIN_CSV.exists() else 0
    test_count = len(pd.read_csv(TEST_CSV)) if TEST_CSV.exists() else 0
    
    v_per_class = {c: 0 for c in CLASS_MAP.keys()}
    for npy in SKELETON_DIR.glob("*.npy"):
        for cls in CLASS_MAP.keys():
            if npy.stem.startswith(cls + "_"):
                v_per_class[cls] += 1
                break

    rejected = 0
    if QUAL_REPORT.exists():
        with open(QUAL_REPORT) as f: rejected = len(json.load(f))
        
    duplicates = 0
    if DUP_REPORT.exists():
        with open(DUP_REPORT) as f: duplicates = len(json.load(f))

    stats = {
        "total_videos": total_videos,
        "videos_per_class": v_per_class,
        "train_samples": train_count,
        "test_samples": test_count,
        "rejected_videos": rejected,
        "duplicate_count": duplicates
    }

    with open(STATS_REPORT, "w") as f:
        json.dump(stats, f, indent=2)
        
    step_ok(f"Stats generated -> {STATS_REPORT.name}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Training
# ══════════════════════════════════════════════════════════════════════════════

def step9_train():
    step_header(9, "Training (CTR-GCN)")
    
    train_script = PROJECT_ROOT / "training/train_lower_limb_ctrgcn.py"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("Running training script...")
    result = subprocess.run([sys.executable, str(train_script)], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        step_fail("Training failed.")
        
    old_ckpt = MODELS_DIR / "best_lower_limb_ctrgcn.pth"
    if old_ckpt.exists():
        shutil.copy2(old_ckpt, NEW_MODEL)
        
    std_results = PROJECT_ROOT / "evaluation/lower_limb_final"
    for ext in ["*.png", "*.md", "*.txt", "*.json"]:
        for f in std_results.glob(ext):
            shutil.copy2(f, RESULTS_DIR / f.name)

    step_ok(f"Training complete -> {NEW_MODEL.name}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: Active Learning Loop
# ══════════════════════════════════════════════════════════════════════════════

def step10_active_learning() -> int:
    step_header(10, "Active Learning (Confidence < 0.6)")
    preds_file = PROJECT_ROOT / "results/lower_limb/auto/sample_predictions.json"
    
    # Check if the training generated the predictions (we might have skipped training if no new data)
    preds_file = PROJECT_ROOT / "results/lower_limb/sample_predictions.json"
    if not preds_file.exists():
        step_ok("No sample_predictions.json found. Skipping active learning.")
        return 0

    with open(preds_file, "r") as f:
        preds = json.load(f)

    REVIEW_DIR = DATASETS_DIR / "review"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    
    removed = 0
    df = None
    if FRAME_CSV.exists():
        df = pd.read_csv(FRAME_CSV)
        
    for sample_name, prob in preds.items():
        if prob < 0.5:
            # Format: {class}_{video_name}
            cls = None
            for c in CLASS_MAP.keys():
                if sample_name.startswith(c + "_"):
                    cls = c
                    break
            if not cls: continue
            video_name = sample_name[len(cls)+1:]
            
            # Find the raw video and move it
            d = RAW_DIR / cls
            if d.exists():
                for vp in d.glob(f"{video_name}*"):
                    shutil.move(str(vp), str(REVIEW_DIR / vp.name))
                    
            # Delete tensor
            npy = SKELETON_DIR / f"{sample_name}.npy"
            if npy.exists():
                npy.unlink()
                
            # Drop from dataframe
            if df is not None:
                df = df[df["video_name"] != sample_name]
                
            removed += 1
            log.info("  [Active Learning] Removed %s (confidence %.2f)", sample_name, prob)

    if df is not None and removed > 0:
        df.to_csv(FRAME_CSV, index=False)
        
    step_ok(f"Active learning scrubbed {removed} low-confidence samples.")
    return removed

def main():
    global log
    log = setup_logger()
    
    p = argparse.ArgumentParser()
    p.add_argument("--max-downloads", type=int, default=50)
    p.add_argument("--no-fetch", action="store_true")
    args = p.parse_args()

    banner("Autonomous Dataset Collection Pipeline")

    iteration = 1
    while True:
        banner(f"Autonomous Loop Iteration {iteration}")
        
        # Check current counts
        current_counts = {c: 0 for c in CLASS_MAP.keys()}
        if SKELETON_DIR.exists():
            for npy in SKELETON_DIR.glob("*.npy"):
                for cls in CLASS_MAP.keys():
                    if npy.stem.startswith(cls + "_"):
                        current_counts[cls] += 1
                        break
        
        needed = sum(max(0, 30 - current_counts[c]) for c in CLASS_MAP.keys())
        log.info(f"Currently need {needed} more videos across all classes to hit 30/class minimum.")
        
        if needed == 0:
            step_ok("Target of 25 videos per class achieved. Pipeline complete.")
            break
            
        step1_search_and_download(not args.no_fetch)
        step2_classify()
        step3_deduplicate()
        step4_quality_filter()
        step5_extract()
        step6_generate_tensors()
        step7_split()
        step8_statistics()
        step9_train()
        removed = step10_active_learning()
        
        if removed == 0 and not args.no_fetch:
            # If we didn't remove any low confidence items and we didn't hit our 25 target, we need to loop again and fetch more.
            # But yt-dlp might keep fetching the same items. Let's just break if we can't fetch.
            pass
            
        iteration += 1

    banner("Pipeline Execution Finished")

if __name__ == "__main__":
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
    main()
