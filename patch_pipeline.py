import sys, re

path = "automation/run_lower_pipeline.py"
with open(path, "r") as f:
    content = f.read()

import_patch = """
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
"""
content = re.sub(r'import cv2\nimport mediapipe as mp\nimport numpy as np', import_patch.strip(), content)

functions_patch = """
def _extract_one_video(video_path: Path, class_name: str, landmarker) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    video_name = video_path.name
    prefixed_video_name = f"{class_name}_{video_name}"
    rows: list[dict] = []
    frame_idx = 0
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

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
    \"\"\"
    Run MediaPipe on new videos only.
    APPENDS new rows to FRAME_CSV (or creates it if missing).
    \"\"\"
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

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        with open(FRAME_CSV, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_cols)
            if not file_exists:
                writer.writeheader()

            for rec in new_videos:
                vp  = Path(rec["source_path"])
                cls = rec["class_name"]
                log.info("  ▸ %-55s class=%s", vp.name[:55], cls)

                rows = _extract_one_video(vp, cls, landmarker)
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
"""

# Replace the two old functions
start_idx = content.find("def _extract_one_video")
end_idx = content.find("# ══════════════════════════════════════════════════════════════════════════════\n# STEP 3")
if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + functions_patch + "\n\n" + content[end_idx:]
    with open(path, "w") as f:
        f.write(content)
    print("Patched!")
else:
    print("Could not find patch bounds")

