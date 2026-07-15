"""
preprocessing/post_extraction_stats.py
──────────────────────────────────────
Reads datasets/upper_limb/upper_limb_frame_labels.csv and generates:
  datasets/upper_limb/extraction_summary.txt
"""
import pandas as pd
from pathlib import Path
import sys

BASE_DIR   = Path(__file__).parent.parent
CSV_PATH   = BASE_DIR / "datasets/upper_limb" / "upper_limb_frame_labels.csv"
OUT_PATH   = BASE_DIR / "datasets/upper_limb" / "extraction_summary.txt"

if not CSV_PATH.exists():
    print(f"[ERROR] CSV not found: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
joint_x_cols = [c for c in df.columns if c.endswith("_x") and c.startswith("joint_")]

total_frames  = len(df)
videos        = df.groupby("video_name")
total_videos  = len(videos)
failed_frames = (df[joint_x_cols[0]] == -1).sum()
det_rate      = (total_frames - failed_frames) / total_frames * 100

lines = []
lines.append("=" * 60)
lines.append("  Upper-Limb Extraction Summary")
lines.append("=" * 60)
lines.append(f"  Total videos processed : {total_videos}")
lines.append(f"  Total frames extracted : {total_frames:,}")
lines.append(f"  Failed detections      : {failed_frames:,}")
lines.append(f"  Detection rate         : {det_rate:.2f}%")
lines.append("")
lines.append("  Per-class video count:")
class_counts = df.drop_duplicates("video_name").groupby("label")["video_name"].count()
for cls, cnt in class_counts.items():
    lines.append(f"    {cls:<22}: {cnt} video(s)")
lines.append("")
lines.append("  Per-video breakdown:")
lines.append(f"  {'Video':<50}  {'Label':<20}  {'Frames':>7}  {'Failed':>7}  {'Det%':>7}")
lines.append("  " + "-" * 95)
for vname, grp in videos:
    n_frames = len(grp)
    n_fail   = (grp[joint_x_cols[0]] == -1).sum()
    rate_v   = (n_frames - n_fail) / n_frames * 100
    lbl      = grp["label"].iloc[0]
    lines.append(f"  {vname[:50]:<50}  {lbl:<20}  {n_frames:>7,}  {n_fail:>7,}  {rate_v:>6.1f}%")
lines.append("=" * 60)

text = "\n".join(lines)
OUT_PATH.write_text(text, encoding="utf-8")
print(text)
print(f"\nSaved to: {OUT_PATH}")
