#!/usr/bin/env python3
"""
tools/balance_and_train.py
End-to-end pipeline:
  Step 1  Rename legacy tensors -> add {class}_ prefix
  Step 2  Extract tensors for all raw videos without one
  Step 3  Download missing videos (yt-dlp) for classes below target
  Step 4  Second extraction pass for newly downloaded videos
  Step 5  Validate & generate dataset_statistics.json
  Step 6  Rebuild stratified train/test CSV splits
  Step 7  Train CTR-GCN
  Step 8  Generate final_training_report.md
Usage:
    python tools/balance_and_train.py
    python tools/balance_and_train.py --skip-download
    python tools/balance_and_train.py --skip-train
"""
from __future__ import annotations
import argparse, csv, json, logging, os, random, re, subprocess, sys, time
from pathlib import Path
import cv2, mediapipe as mp, numpy as np

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATASETS_DIR    = PROJECT_ROOT / "datasets" / "lower_limb"
RAW_DIR         = DATASETS_DIR / "raw"
SKELETON_DIR    = DATASETS_DIR / "skeletons"
RESULTS_DIR     = PROJECT_ROOT / "evaluation" / "lower_limb_final"
MODELS_DIR      = PROJECT_ROOT / "models"
LOGS_DIR        = PROJECT_ROOT / "logs"
TRAIN_CSV       = DATASETS_DIR / "train_labels.csv"
TEST_CSV        = DATASETS_DIR / "test_labels.csv"
MEDIAPIPE_MODEL = MODELS_DIR / "pose_landmarker_full.task"

TARGET     = 40
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
CLASS_MAP: dict[str, int] = {
    "ankle":8, "calf":1, "hamstring":5, "heel_slide":6,
    "hip":4, "knee":7, "leg_raise":2, "quadriceps":0, "toes":3,
}
LOWER_LIMB_JOINTS = [23,24,25,26,27,28,29,30,31,32]

QUERIES: dict[str, list[str]] = {
    "ankle":      ["ankle pumps physiotherapy","ankle dorsiflexion exercise",
                   "ankle inversion exercise","ankle rehabilitation exercise"],
    "calf":       ["calf stretch physiotherapy","calf strengthening exercise",
                   "seated calf stretch","standing calf stretch"],
    "hamstring":  ["hamstring stretch physiotherapy","seated hamstring stretch",
                   "hamstring rehabilitation exercise","supine hamstring stretch"],
    "heel_slide": ["heel slide exercise","heel slide physiotherapy",
                   "post surgery heel slide","supine heel slides exercise"],
    "hip":        ["hip abduction exercise","hip rehabilitation exercise",
                   "hip strengthening physiotherapy","sidelying hip abduction"],
    "knee":       ["knee extension exercise","knee rehabilitation exercise",
                   "knee strengthening physiotherapy","knee flexion physiotherapy"],
    "leg_raise":  ["straight leg raise exercise","supine leg raise physiotherapy",
                   "leg raise strengthening","straight leg raise rehabilitation"],
    "quadriceps": ["quadriceps sets physiotherapy","quad set exercise",
                   "quadriceps strengthening exercise","isometric quadriceps exercise"],
    "toes":       ["toe raise exercise","dorsiflexion exercise",
                   "toe raise physiotherapy","toe curls exercise"],
}

LEGACY_KW: dict[str, list[str]] = {
    "ankle":      ["ankle","dorsiflexion","plantarflexion","inversion","eversion"],
    "calf":       ["calf","gastrocnemius"],
    "hamstring":  ["hamstring"],
    "heel_slide": ["heel slide","heel_slide","heelslide","trans-abdominal-heel"],
    "hip":        ["hip bridge","hip abduction","hip shifting","supine hip","kneeling hip flexor","hip excercise"],
    "knee":       ["hinged knee","small knee","cook","knee flexion","knee extension","knee hold","knee variation"],
    "leg_raise":  ["leg raise","leg_lift","leg pull","straight leg","staright leg"],
    "quadriceps": ["quad","quadriceps","asmr quads","activated squat"],
    "toes":       ["toe","big toe"],
}

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("balance_train")

def banner(msg):
    log.info("\n" + "=" * 65)
    log.info("  %s", msg)
    log.info("=" * 65)

def infer_class(stem: str):
    s = stem.lower()
    for cls, kws in LEGACY_KW.items():
        for kw in kws:
            if kw in s:
                return cls
    return None

def count_tensors() -> dict[str,int]:
    c = {cls:0 for cls in CLASS_MAP}
    for npy in SKELETON_DIR.glob("*.npy"):
        for cls in CLASS_MAP:
            if npy.name.startswith(cls+"_"):
                c[cls] += 1; break
    return c

def extract_tensor(vpath: Path, dest: Path) -> bool:
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv
    if not MEDIAPIPE_MODEL.exists():
        log.error("Model missing: %s", MEDIAPIPE_MODEL); return False
    opts = mpv.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
        running_mode=mpv.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.25, min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25)
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened(): return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seq, fidx = [], 0
    try:
        with mpv.PoseLandmarker.create_from_options(opts) as lm:
            while True:
                ret, bgr = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = lm.detect_for_video(mpi, int((fidx/fps)*1000))
                fidx += 1
                if res.pose_landmarks:
                    lms = res.pose_landmarks[0]
                    row = np.zeros((10,4), dtype=np.float32)
                    for i,jid in enumerate(LOWER_LIMB_JOINTS):
                        idx = jid-23
                        l   = lms[idx] if idx < len(lms) else lms[0]
                        row[i] = [l.x, l.y, l.z, l.visibility]
                    seq.append(row)
    finally:
        cap.release()
    if fidx == 0 or len(seq)/fidx < 0.25: return False
    s = np.array(seq, dtype=np.float32)
    T = s.shape[0]
    if T != 300:
        xo, xn = np.linspace(0,1,T), np.linspace(0,1,300)
        s300 = np.zeros((300,10,4), dtype=np.float32)
        for i in range(10):
            for j in range(4):
                s300[:,i,j] = np.interp(xn, xo, s[:,i,j])
        s = s300
    t = np.transpose(s,(2,0,1))[:,:,:,np.newaxis]
    np.save(str(dest), t.astype(np.float32))
    return True

def step1_rename() -> int:
    banner("STEP 1 — Rename legacy tensors")
    renamed = 0
    for npy in sorted(SKELETON_DIR.glob("*.npy")):
        stem = npy.stem
        if any(stem.startswith(c+"_") for c in CLASS_MAP): continue
        cls = infer_class(stem)
        if cls is None:
            log.warning("  Unclassified: %s", npy.name); continue
        new = SKELETON_DIR / f"{cls}_{stem}.npy"
        if new.exists():
            npy.unlink(); continue
        npy.rename(new)
        log.info("  %s => %s_%s.npy", npy.name, cls, stem)
        renamed += 1
    log.info("  Renamed %d", renamed); return renamed

def step2_extract() -> int:
    banner("STEP 2/4 — Extract tensors from raw videos")
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    gen = skipped = failed = 0
    for cls in CLASS_MAP:
        d = RAW_DIR / cls
        if not d.exists(): continue
        for vp in sorted(d.iterdir()):
            if vp.suffix.lower() not in VIDEO_EXTS: continue
            stem = re.sub(r"\.f\d+$","", re.sub(r"f\d+$","", vp.stem)).strip()
            dest = SKELETON_DIR / f"{cls}_{stem}.npy"
            if dest.exists():
                skipped += 1; continue
            log.info("  [%s] %s", cls, vp.name)
            if extract_tensor(vp, dest): gen += 1
            else: failed += 1; log.warning("  Failed: %s", vp.name)
    log.info("  Gen=%d Skip=%d Fail=%d", gen, skipped, failed)
    return gen

def step3_download(skip: bool) -> None:
    banner("STEP 3 — Download missing classes")
    if skip:
        log.info("  --skip-download"); return
    log.info("  Updating yt-dlp …")
    subprocess.run([sys.executable,"-m","pip","install","-U","yt-dlp","--quiet"], check=False)
    import yt_dlp
    counts = count_tensors()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    failures, stats = [], {"attempts":0,"success":0,"failed":0,"403":0}
    for cls, queries in QUERIES.items():
        needed = TARGET - counts.get(cls,0)
        if needed <= 0:
            log.info("  [%s] OK (%d). Skip.", cls, counts.get(cls,0)); continue
        dest_dir = RAW_DIR / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        log.info("  [%s] Need %d more.", cls, needed)
        ydl_opts = {
            "format": "b[ext=mp4][height<=720]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(dest_dir / "%(title)s.%(ext)s"),
            "match_filter": yt_dlp.utils.match_filter_func("duration >= 10 & duration <= 300"),
            "ignoreerrors": True, "quiet": True, "no_warnings": True,
        }
        collected = 0
        for q in queries:
            if collected >= needed: break
            for attempt, wait in enumerate([5,15,30]):
                try:
                    stats["attempts"] += 1
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(f"ytsearch{needed-collected}:{q}", download=True)
                    if info and "entries" in info:
                        for e in (info["entries"] or []):
                            if not e: continue
                            vp = dest_dir / f"{e.get('title','x')}.{e.get('ext','mp4')}"
                            if vp.exists() and vp.stat().st_size > 500_000:
                                collected += 1; stats["success"] += 1
                    break
                except Exception as err:
                    es = str(err)
                    log.warning("  Attempt %d [%s]: %s", attempt+1, q, es[:100])
                    if "403" in es: stats["403"] += 1
                    failures.append({"q":q,"attempt":attempt+1,"error":es})
                    if attempt < 2: time.sleep(wait)
                    else: stats["failed"] += 1
    with open(LOGS_DIR/"download_failures.json","w") as f: json.dump(failures,f,indent=2)
    with open(PROJECT_ROOT/"download_statistics.json","w") as f: json.dump(stats,f,indent=2)
    log.info("  %s", stats)

def step5_validate() -> dict[str,int]:
    banner("STEP 5 — Validate")
    counts = count_tensors()
    raw_c  = {}
    for cls in CLASS_MAP:
        d = RAW_DIR/cls
        raw_c[cls] = sum(1 for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS) if d.exists() else 0
    stats = {"total": sum(counts.values()), "tensors": counts, "videos": raw_c,
             "target": TARGET, "below": [c for c,n in counts.items() if n<TARGET]}
    with open(PROJECT_ROOT/"dataset_statistics.json","w") as f: json.dump(stats,f,indent=2)
    log.info("  Total tensors: %d", stats["total"])
    for c in sorted(CLASS_MAP.keys()):
        n = counts.get(c,0)
        log.info("    %-12s  %3d  %s", c, n, "OK" if n>=TARGET else f"need +{TARGET-n}")
    if stats["total"] == 0:
        log.error("No tensors! Aborting."); sys.exit(1)
    return counts

def step6_splits(counts) -> None:
    banner("STEP 6 — Rebuild train/test splits")
    buckets: dict[int,list] = {}
    for npy in sorted(SKELETON_DIR.glob("*.npy")):
        cls = next((c for c in CLASS_MAP if npy.name.startswith(c+"_")), None)
        if cls is None: continue
        buckets.setdefault(CLASS_MAP[cls],[]).append((npy.stem, CLASS_MAP[cls]))
    random.seed(42)
    train_r, test_r = [], []
    for label, items in buckets.items():
        random.shuffle(items)
        sp = max(1, int(0.8*len(items)))
        train_r.extend(items[:sp]); test_r.extend(items[sp:])
    random.shuffle(train_r); random.shuffle(test_r)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in [(TRAIN_CSV, train_r),(TEST_CSV, test_r)]:
        with open(path,"w",newline="") as f:
            w = csv.writer(f); w.writerow(["sample_name","label"]); w.writerows(rows)
    log.info("  Train=%d  Test=%d", len(train_r), len(test_r))

def step7_train(skip: bool) -> None:
    banner("STEP 7 — Train CTR-GCN")
    if skip:
        log.info("  --skip-train"); return
    script = PROJECT_ROOT/"training"/"train_lower_limb_ctrgcn.py"
    if not script.exists():
        log.error("Not found: %s", script); return
    r = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), check=False)
    if r.returncode: log.error("Training failed (code %d)", r.returncode)
    else: log.info("  Training complete.")

def step8_report(counts: dict[str,int]) -> None:
    banner("STEP 8 — Generate final_training_report.md")
    metrics = {}
    for cand in [RESULTS_DIR/"metrics.json",
                 PROJECT_ROOT/"results"/"lower_limb"/"metrics.json",
                 PROJECT_ROOT/"results"/"lower_limb"/"auto"/"metrics.json"]:
        if cand.exists():
            with open(cand) as f: metrics=json.load(f); break
    def rc(p): return sum(1 for _ in open(p))-1 if p.exists() else "N/A"
    raw_c={cls:(sum(1 for p in (RAW_DIR/cls).iterdir() if p.suffix.lower() in VIDEO_EXTS) if (RAW_DIR/cls).exists() else 0) for cls in CLASS_MAP}
    rows=""
    for cls in sorted(CLASS_MAP.keys()):
        rv=raw_c.get(cls,0); tc=counts.get(cls,0)
        rows+=f"| {cls:<12} | {rv:>6} | {tc:>7} | {'OK' if tc>=TARGET else f'need +{TARGET-tc}'} |\n"
    perf="*Training metrics not yet available.*\n"
    if metrics:
        perf=f"""| Metric | Value |
|--------|-------|
| Accuracy   | {metrics.get('accuracy','N/A')} |
| Macro F1   | {metrics.get('macro_f1','N/A')} |
| Top-3 Acc  | {metrics.get('top3_accuracy','N/A')} |
| Best Epoch | {metrics.get('best_epoch','N/A')} |
"""
        pf=metrics.get("per_class_f1",{})
        if pf:
            perf += "\n### Per-class F1\n| Class | F1 |\n|-------|----|\n"
            for c,f in sorted(pf.items()): perf+=f"| {c} | {f:.4f} |\n"
    report=f"""# Lower-Limb CTR-GCN — Final Training Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
| Class        | Videos | Tensors | Status |
|--------------|--------|---------|--------|
{rows}
Total tensors: {sum(counts.values())}  |  Train: {rc(TRAIN_CSV)}  |  Test: {rc(TEST_CSV)}

## Training Config
epochs=250, batch=32, lr=0.001, AdamW, CosineAnnealingLR,
label_smoothing=0.1, early_stop_patience=30,
WeightedRandomSampler, Weighted CrossEntropy

## Evaluation
{perf}
## Artefacts
- `models/best_lower_limb_final.pth`
- `results/final/confusion_matrix.png`
- `results/final/accuracy_curve.png`
- `results/final/loss_curve.png`
- `results/final/classification_report.txt`
- `results/final/metrics.json`
"""
    out = PROJECT_ROOT/"final_training_report.md"
    out.write_text(report)
    log.info("  Report: %s", out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train",    action="store_true")
    args = parser.parse_args()
    banner("Lower-Limb Dataset Balance & Train Pipeline")
    step1_rename()
    step2_extract()
    step3_download(skip=args.skip_download)
    step2_extract()   # second pass for newly downloaded
    counts = step5_validate()
    step6_splits(counts)
    step7_train(skip=args.skip_train)
    step8_report(counts)
    banner("Pipeline Complete")

if __name__ == "__main__":
    main()
