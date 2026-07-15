#!/usr/bin/env python3
"""
training/lower_limb/train_final.py
────────────────────────────────────────────────────────────────────────────
Production-grade Lower-Limb CTR-GCN Training Pipeline
Apple Silicon (MPS) optimised

Steps executed automatically:
  1. Repository validation
  2. Tensor validation & in-place repair
  3. Dataset statistics + class_distribution.png
  4/5. Training configuration + Apple Silicon optimisation
  6. Training (250 epochs, early-stop, per-10-epoch checkpoints)
  7. Full evaluation suite (confusion matrix, ROC, per-class F1, curves)
  8. training_report.md
  9. ONNX export + deployment_metadata.json
  10. Final summary printout

Usage:
    python training/lower_limb/train_final.py
    python training/lower_limb/train_final.py --epochs 50  # quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_curve, auc
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[2]
SKELETON_DIR  = BASE_DIR / "datasets/lower_limb/skeletons"
TRAIN_CSV     = BASE_DIR / "datasets/lower_limb/train_labels.csv"
TEST_CSV      = BASE_DIR / "datasets/lower_limb/test_labels.csv"
MODELS_DIR    = BASE_DIR / "models/lower_limb"
EVAL_DIR      = BASE_DIR / "evaluation/lower_limb_final"
ONNX_PATH     = MODELS_DIR / "best_model.onnx"

BEST_MODEL    = MODELS_DIR / "best_model.pth"
LAST_MODEL    = MODELS_DIR / "last_model.pth"

sys.path.insert(0, str(BASE_DIR))
from preprocessing.lower_limb.loader import (
    CLASS_NAMES, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
    PhysioSkeletonDataset,
)
from preprocessing.lower_limb.graph import LowerLimbGraph
from model.ctrgcn import Model

# ── Class mapping ──────────────────────────────────────────────────────────────
CLASS_MAP = {
    "ankle": 8, "calf": 1, "hamstring": 5, "heel_slide": 6,
    "hip": 4,   "knee": 7, "leg_raise": 2, "quadriceps": 0, "toes": 3,
}
NUM_CLASSES = 9

EXPECTED_SHAPE = (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)

# ── Logger ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "training_final.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("train_final")

def banner(msg: str) -> None:
    log.info("\n" + "=" * 70)
    log.info("  %s", msg)
    log.info("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Repository Validation
# ══════════════════════════════════════════════════════════════════════════════

def step1_validate_repo() -> dict:
    banner("STEP 1 — Repository Validation")
    issues = []

    critical_paths = {
        "Skeleton dir":     SKELETON_DIR,
        "train CSV":        TRAIN_CSV,
        "test CSV":         TEST_CSV,
        "CTR-GCN model":   BASE_DIR / "model/ctrgcn.py",
        "MediaPipe model": BASE_DIR / "models/pose_landmarker_full.task",
        "Lower-limb graph":BASE_DIR / "preprocessing/lower_limb/graph.py",
        "DataLoader":      BASE_DIR / "preprocessing/lower_limb/loader.py",
    }
    for name, p in critical_paths.items():
        if p.exists():
            log.info("  ✅  %-25s  %s", name, p.relative_to(BASE_DIR))
        else:
            log.error("  ❌  %-25s  MISSING: %s", name, p)
            issues.append(f"MISSING: {p}")

    if not SKELETON_DIR.exists() or not any(SKELETON_DIR.glob("*.npy")):
        log.error("  No .npy tensors found in %s", SKELETON_DIR)
        sys.exit(1)

    npy_files = list(SKELETON_DIR.glob("*.npy"))
    log.info("  Total tensors on disk: %d", len(npy_files))

    if issues:
        log.warning("  %d non-critical issues found. Continuing.", len(issues))

    return {"npy_files": npy_files, "issues": issues}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tensor Validation & Repair
# ══════════════════════════════════════════════════════════════════════════════

def _fix_tensor(arr: np.ndarray, path: Path) -> tuple[np.ndarray, list[str]]:
    """Repair a tensor to shape (4,300,10,1) float32. Returns (fixed, repairs_applied)."""
    repairs = []

    # Fix dtype
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        repairs.append("dtype→float32")

    # Handle legacy (T,V,C) shape
    if arr.ndim == 3 and arr.shape == (300, 10, 4):
        arr = np.transpose(arr, (2, 0, 1))[:, :, :, np.newaxis]
        repairs.append("transposed (T,V,C)→(C,T,V,M)")

    # Handle (C,T,V) missing M dim
    if arr.ndim == 3 and arr.shape == (4, 300, 10):
        arr = arr[:, :, :, np.newaxis]
        repairs.append("added M dim")

    # Fix NaN / Inf
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        repairs.append("NaN/Inf→0")

    # Fix visibility clamp [0,1]
    arr[3] = np.clip(arr[3], 0.0, 1.0)

    # Fix frame count mismatch via interpolation
    if arr.ndim == 4 and arr.shape[1] != 300:
        T_in = arr.shape[1]
        x_old = np.linspace(0, 1, T_in)
        x_new = np.linspace(0, 1, 300)
        arr_new = np.zeros((4, 300, 10, 1), dtype=np.float32)
        for c in range(4):
            for v in range(10):
                arr_new[c, :, v, 0] = np.interp(x_new, x_old, arr[c, :, v, 0])
        arr = arr_new
        repairs.append(f"frames {T_in}→300")

    return arr, repairs


def step2_validate_tensors(npy_files: list[Path]) -> dict:
    banner("STEP 2 — Tensor Validation & Repair")

    ok = repaired = corrupted = 0
    repair_log = []
    corrupt_list = []

    for p in npy_files:
        try:
            arr = np.load(str(p))
            fixed, repairs = _fix_tensor(arr, p)

            if fixed.shape != EXPECTED_SHAPE:
                log.error("  CORRUPT (unfixable) %s  shape=%s", p.name, arr.shape)
                corrupted += 1
                corrupt_list.append(str(p.name))
                continue

            if repairs:
                np.save(str(p), fixed)
                repair_log.append({"file": p.name, "repairs": repairs})
                repaired += 1
                log.info("  REPAIRED  %s  [%s]", p.name, ", ".join(repairs))
            else:
                ok += 1

        except Exception as e:
            log.error("  CORRUPT (load error) %s: %s", p.name, e)
            corrupted += 1
            corrupt_list.append(p.name)

    log.info("  Tensors OK=%d  Repaired=%d  Corrupt=%d", ok, repaired, corrupted)

    if corrupted / max(len(npy_files), 1) > 0.3:
        log.error("Critical: >30%% tensors corrupt. Aborting.")
        sys.exit(1)

    return {"ok": ok, "repaired": repaired, "corrupted": corrupted,
            "corrupt_list": corrupt_list, "repair_log": repair_log}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Dataset Statistics & Split Rebuild
# ══════════════════════════════════════════════════════════════════════════════

def step3_build_stats_and_splits(npy_files: list[Path], val_info: dict) -> dict:
    banner("STEP 3 — Dataset Statistics & Split Rebuild")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Classify every valid tensor
    class_buckets: dict[int, list[str]] = {}
    unclassified = []

    for p in npy_files:
        if p.name in val_info["corrupt_list"]:
            continue
        stem = p.stem
        cls  = next((c for c in CLASS_MAP if stem.startswith(c + "_")), None)
        if cls is None:
            unclassified.append(stem)
            continue
        label = CLASS_MAP[cls]
        class_buckets.setdefault(label, []).append(stem)

    if unclassified:
        log.warning("  %d tensors could not be classified by prefix: %s...",
                    len(unclassified), unclassified[:3])

    # Deduplicate within each class
    dupes_removed = 0
    for label in class_buckets:
        seen = {}
        clean = []
        for stem in class_buckets[label]:
            key = stem.lower().strip()
            if key not in seen:
                seen[key] = True
                clean.append(stem)
            else:
                dupes_removed += 1
        class_buckets[label] = clean

    # Build stratified 80/20 split
    random.seed(42)
    train_rows, test_rows = [], []
    class_dist = {}

    for label, stems in sorted(class_buckets.items()):
        cls_name = CLASS_NAMES.get(label, str(label))
        random.shuffle(stems)
        split = max(1, int(0.8 * len(stems)))
        train_rows.extend([(s, label) for s in stems[:split]])
        test_rows.extend( [(s, label) for s in stems[split:]])
        class_dist[cls_name] = len(stems)
        log.info("  %-12s  total=%-3d  train=%-3d  test=%d",
                 cls_name, len(stems), split, len(stems) - split)

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in [(TRAIN_CSV, train_rows), (TEST_CSV, test_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_name", "label"])
            w.writerows(rows)

    log.info("  Train=%d  Test=%d  Dupes removed=%d",
             len(train_rows), len(test_rows), dupes_removed)

    # Stats JSON
    stats = {
        "total_tensors":    len(npy_files) - val_info["corrupted"],
        "train_samples":    len(train_rows),
        "test_samples":     len(test_rows),
        "classes":          class_dist,
        "duplicates_removed": dupes_removed,
        "repaired_tensors": val_info["repaired"],
        "corrupted_tensors": val_info["corrupted"],
        "unclassified":     len(unclassified),
    }
    with open(BASE_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Class distribution chart
    names  = list(class_dist.keys())
    counts = list(class_dist.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, counts, color=plt.cm.tab10.colors[:len(names)])
    ax.set_title("Lower-Limb Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tensor Count")
    ax.set_xlabel("Exercise Class")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "class_distribution.png", dpi=150)
    plt.close()

    log.info("  class_distribution.png saved")
    return stats

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 + 5 — Training Setup & Apple Silicon Optimisation
# ══════════════════════════════════════════════════════════════════════════════

def step45_setup(args) -> dict:
    banner("STEP 4+5 — Training Config & Apple Silicon Setup")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("  Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("  Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        log.info("  Device: CPU")

    # Num-workers
    num_workers = 4 if device.type != "cuda" else min(8, os.cpu_count() or 4)
    pin_memory  = (device.type == "cuda")
    log.info("  num_workers=%d  pin_memory=%s", num_workers, pin_memory)

    # AMP
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    log.info("  AMP: %s", "ENABLED" if use_amp else "disabled (MPS/CPU)")

    cfg = {
        "device":       device,
        "num_workers":  num_workers,
        "pin_memory":   pin_memory,
        "use_amp":      use_amp,
        "scaler":       scaler,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "weight_decay": 1e-4,
        "grad_clip":    1.0,
        "patience":     30,
        "grad_accum":   args.grad_accum,
    }
    log.info("  epochs=%d  batch=%d  lr=%g  grad_accum=%d",
             cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["grad_accum"])
    return cfg

# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, criterion, device,
                scaler, grad_clip, grad_accum) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    optimiser.zero_grad(set_to_none=True)

    for step, (data, labels, _) in enumerate(loader, 1):
        data   = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss   = criterion(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
        else:
            logits = model(data)
            loss   = criterion(logits, labels) / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, list, list]:
    model.eval()
    all_preds, all_targets = [], []
    for data, labels, _ in loader:
        data   = data.to(device, non_blocking=True)
        preds  = model(data).argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)
    return acc, all_preds, all_targets


@torch.no_grad()
def get_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_targets = [], []
    for data, labels, _ in loader:
        logits = model(data.to(device, non_blocking=True))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(labels.numpy())
    return np.vstack(all_probs), np.array(all_targets)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def step6_train(cfg: dict) -> dict:
    banner("STEP 6 — Training")

    # Datasets
    train_ds = PhysioSkeletonDataset(str(TRAIN_CSV), str(SKELETON_DIR), augment=True)
    test_ds  = PhysioSkeletonDataset(str(TEST_CSV),  str(SKELETON_DIR), augment=False)

    # WeightedRandomSampler
    dist = train_ds.class_distribution
    total_samples = sum(dist.values())
    sample_wts = [1.0 / max(dist.get(s["label"], 1), 1) for s in train_ds.samples]
    sampler    = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

    device   = cfg["device"]
    nw       = cfg["num_workers"]
    pin      = cfg["pin_memory"]
    pw       = (nw > 0)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=nw, pin_memory=pin,
                              persistent_workers=pw, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=pin,
                              persistent_workers=pw, drop_last=False)

    log.info("  Train=%d  Test=%d", len(train_ds), len(test_ds))

    # Class-weighted loss
    cw = torch.zeros(NUM_CLASSES, device=device)
    for i in range(NUM_CLASSES):
        cnt = dist.get(i, 0)
        cw[i] = total_samples / (NUM_CLASSES * cnt) if cnt > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    # Model
    graph = LowerLimbGraph()
    model = Model(
        num_class=NUM_CLASSES, num_point=EXPECTED_V,
        num_person=EXPECTED_M, in_channels=EXPECTED_C, graph=graph,
    ).to(device)

    # torch.compile (optional, Python 3.12+ partial support)
    # Disabled for MPS since inductor doesn't support it yet.
    # try:
    #     model = torch.compile(model)
    #     log.info("  torch.compile: ENABLED")
    # except Exception:
    #     log.info("  torch.compile: not available, skipping")
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg["epochs"], eta_min=1e-6)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0
    train_losses, train_accs, val_accs = [], [], []
    t_start = time.perf_counter()

    log.info("  %-6s  %-10s  %-12s  %-12s  %-10s",
             "Epoch", "Loss", "Train%", "Val%", "LR")
    log.info("  " + "-" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimiser, criterion, device,
            cfg["scaler"], cfg["grad_clip"], cfg["grad_accum"]
        )
        val_acc, _, _ = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        cur_lr = scheduler.get_last_lr()[0]
        log.info("  %-6d  %-10.4f  %-12.2f  %-12.2f  %-10.6f",
                 epoch, tr_loss, tr_acc * 100, val_acc * 100, cur_lr)

        # Best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save({
                "epoch": epoch, "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "num_class": NUM_CLASSES, "num_point": EXPECTED_V,
                "num_person": EXPECTED_M, "in_channels": EXPECTED_C,
            }, BEST_MODEL)
            log.info("  ★  Best checkpoint  val_acc=%.2f%%  epoch=%d",
                     val_acc * 100, epoch)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                log.info("  Early stopping at epoch %d", epoch)
                break

        # Every-10-epoch checkpoint
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       MODELS_DIR / f"epoch_{epoch}.pth")

    # Save last model
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, LAST_MODEL)

    elapsed = time.perf_counter() - t_start
    log.info("  Training complete. Elapsed: %.1fs", elapsed)

    return {
        "model": model, "train_loader": train_loader, "test_loader": test_loader,
        "train_losses": train_losses, "train_accs": train_accs, "val_accs": val_accs,
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "elapsed": elapsed, "device": device,
        "train_ds": train_ds, "test_ds": test_ds,
    }

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Full Evaluation Suite
# ══════════════════════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def step7_evaluate(results: dict) -> dict:
    banner("STEP 7 — Evaluation")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _style()

    model      = results["model"]
    device     = results["device"]
    test_loader= results["test_loader"]
    test_ds    = results["test_ds"]

    # Load best checkpoint
    ckpt = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    # Handle torch.compile _orig_mod wrapper
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=False)

    final_acc, preds, targets = evaluate(model, test_loader, device)
    probs, _                  = get_probs(model, test_loader, device)

    present_in_test = sorted(set(targets))
    target_names    = [CLASS_NAMES.get(c, str(c)) for c in present_in_test]

    # Metrics
    macro_f1    = f1_score(targets, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    precision   = precision_score(targets, preds, average="macro", zero_division=0)
    recall      = recall_score(targets, preds, average="macro",    zero_division=0)
    per_cls_f1  = f1_score(targets, preds, average=None, labels=present_in_test, zero_division=0)

    # Top-3 accuracy
    top3_correct = 0
    for i, t in enumerate(targets):
        top3 = np.argsort(probs[i])[-3:]
        if t in top3:
            top3_correct += 1
    top3_acc = top3_correct / max(len(targets), 1)

    metrics = {
        "accuracy":     round(float(final_acc),    4),
        "macro_f1":     round(float(macro_f1),     4),
        "weighted_f1":  round(float(weighted_f1),  4),
        "precision":    round(float(precision),     4),
        "recall":       round(float(recall),        4),
        "top3_accuracy":round(float(top3_acc),      4),
        "best_val_acc": round(float(results["best_val_acc"]), 4),
        "best_epoch":   int(results["best_epoch"]),
        "per_class_f1": {
            CLASS_NAMES.get(c, str(c)): round(float(v), 4)
            for c, v in zip(present_in_test, per_cls_f1)
        },
    }
    with open(EVAL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report
    report_str = classification_report(
        targets, preds, labels=present_in_test,
        target_names=target_names, zero_division=0
    )
    (EVAL_DIR / "classification_report.txt").write_text(report_str)
    log.info("  Accuracy=%.2f%%  MacroF1=%.4f  Top3=%.2f%%",
             final_acc*100, macro_f1, top3_acc*100)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(targets, preds, labels=present_in_test)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Lower-Limb CTR-GCN", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ── Loss & Accuracy Curves ────────────────────────────────────────────────
    epochs_x = list(range(1, len(results["train_losses"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_x, results["train_losses"], label="Train Loss", color="#E74C3C")
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

    ax2.plot(epochs_x, [a*100 for a in results["train_accs"]],
             label="Train Acc", color="#2ECC71")
    ax2.plot(epochs_x, [a*100 for a in results["val_accs"]],
             label="Val Acc", color="#3498DB")
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend()

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "accuracy_curve.png", dpi=150)
    plt.savefig(EVAL_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # ── Per-class F1 bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    f1_vals = [metrics["per_class_f1"].get(n, 0) for n in target_names]
    colors  = ["#27AE60" if v >= 0.7 else "#F39C12" if v >= 0.5 else "#E74C3C"
               for v in f1_vals]
    ax.bar(target_names, f1_vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, linestyle="--", color="gray", alpha=0.5, label="0.70 threshold")
    ax.set_title("Per-class F1 Score", fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("F1 Score")
    for i, (name, v) in enumerate(zip(target_names, f1_vals)):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(EVAL_DIR / "per_class_f1.png", dpi=150)
    plt.close()

    # ── ROC Curves ───────────────────────────────────────────────────────────
    from sklearn.preprocessing import label_binarize
    all_classes = list(range(NUM_CLASSES))
    y_bin = label_binarize(targets, classes=all_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls_idx in enumerate(present_in_test):
        if y_bin[:, cls_idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], probs[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES.get(cls_idx,str(cls_idx))} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — Lower-Limb CTR-GCN", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(EVAL_DIR / "roc_curve.png", dpi=150)
    plt.close()

    log.info("  All evaluation artefacts saved → %s", EVAL_DIR)
    return metrics

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — training_report.md
# ══════════════════════════════════════════════════════════════════════════════

def step8_report(results: dict, metrics: dict, stats: dict) -> None:
    banner("STEP 8 — Training Report")
    import psutil, platform

    ram_gb = psutil.virtual_memory().total / 1e9

    per_cls_rows = "\n".join(
        f"| {cls:<12} | {f1:.4f} |"
        for cls, f1 in metrics["per_class_f1"].items()
    )

    report = f"""# Lower-Limb CTR-GCN — Final Training Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Summary
| Metric | Value |
|--------|-------|
| Total tensors | {stats["total_tensors"]} |
| Train samples | {stats["train_samples"]} |
| Test samples  | {stats["test_samples"]} |
| Repaired tensors | {stats["repaired_tensors"]} |
| Duplicates removed | {stats["duplicates_removed"]} |

## Training
| Parameter | Value |
|-----------|-------|
| Epochs run | {results["best_epoch"]} (best) / {len(results["train_losses"])} (total) |
| Best val accuracy | {metrics["best_val_acc"]*100:.2f}% |
| Training duration | {results["elapsed"]/60:.1f} min |
| Device | {results["device"]} |
| RAM | {ram_gb:.1f} GB |
| Platform | {platform.machine()} / {platform.system()} |

## Evaluation
| Metric | Value |
|--------|-------|
| Accuracy | {metrics["accuracy"]*100:.2f}% |
| Macro F1 | {metrics["macro_f1"]:.4f} |
| Weighted F1 | {metrics["weighted_f1"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| Top-3 Accuracy | {metrics["top3_accuracy"]*100:.2f}% |

## Per-class F1
| Class | F1 |
|-------|----|
{per_cls_rows}

## Artefacts
- `models/lower_limb/best_model.pth`
- `models/lower_limb/last_model.pth`
- `models/lower_limb/best_model.onnx`
- `evaluation/lower_limb_final/confusion_matrix.png`
- `evaluation/lower_limb_final/accuracy_curve.png`
- `evaluation/lower_limb_final/loss_curve.png`
- `evaluation/lower_limb_final/per_class_f1.png`
- `evaluation/lower_limb_final/roc_curve.png`
- `evaluation/lower_limb_final/classification_report.txt`
- `evaluation/lower_limb_final/metrics.json`
- `evaluation/lower_limb_final/class_distribution.png`
- `evaluation/lower_limb_final/deployment_metadata.json`
"""
    (BASE_DIR / "training_report.md").write_text(report)
    log.info("  training_report.md saved")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — ONNX Export & Deployment Metadata
# ══════════════════════════════════════════════════════════════════════════════

def step9_export(results: dict) -> None:
    banner("STEP 9 — ONNX Export & Deployment Metadata")
    model  = results["model"]
    device = results["device"]

    # Load best weights
    ckpt = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=False)
    model.eval()

    # ONNX export (CPU for portability)
    model_cpu = model.to("cpu")
    dummy     = torch.zeros(1, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)
    try:
        torch.onnx.export(
            model_cpu, dummy, str(ONNX_PATH),
            input_names=["skeleton"],
            output_names=["logits"],
            dynamic_axes={"skeleton": {0: "batch"}},
            opset_version=17,
        )
        log.info("  ONNX exported: %s", ONNX_PATH)
    except Exception as e:
        log.warning("  ONNX export failed: %s", e)

    # Deployment metadata
    joint_map = {str(23+i): name for i, name in enumerate([
        "left_hip","right_hip","left_knee","right_knee",
        "left_ankle","right_ankle","left_heel","right_heel",
        "left_foot_index","right_foot_index"
    ])}
    meta = {
        "class_names":          {str(v): k for k, v in CLASS_MAP.items()},
        "joint_mapping":        joint_map,
        "num_classes":          NUM_CLASSES,
        "input_shape":          [1, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M],
        "output_shape":         [1, NUM_CLASSES],
        "channels":             {"0":"x","1":"y","2":"z","3":"visibility"},
        "normalization":        "none (raw MediaPipe normalized coordinates)",
        "mediapipe_model":      "pose_landmarker_full.task",
        "framework":            "PyTorch + ONNX",
        "target_hardware":      "Apple Silicon MPS / CUDA / CPU",
    }
    with open(EVAL_DIR / "deployment_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("  deployment_metadata.json saved")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Final Summary
# ══════════════════════════════════════════════════════════════════════════════

def step10_summary(results: dict, metrics: dict, stats: dict) -> None:
    banner("STEP 10 — Final Summary")
    print("\n" + "=" * 50)
    print(f"  Dataset Size  :  {stats['total_tensors']} tensors")
    print(f"  Train Samples :  {stats['train_samples']}")
    print(f"  Test Samples  :  {stats['test_samples']}")
    print(f"  Best Accuracy :  {metrics['best_val_acc']*100:.2f}%")
    print(f"  Best Macro F1 :  {metrics['macro_f1']:.4f}")
    print(f"  Top-3 Accuracy:  {metrics['top3_accuracy']*100:.2f}%")
    print(f"  Training Time :  {results['elapsed']/60:.1f} min")
    print(f"  Best Epoch    :  {results['best_epoch']}")
    print(f"  Model Location:  {BEST_MODEL}")
    print("=" * 50 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=250)
    parser.add_argument("--batch-size", type=int,   default=16,  dest="batch_size")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--grad-accum", type=int,   default=2,   dest="grad_accum",
                        help="Gradient accumulation steps (default 2 for MPS memory)")
    parser.add_argument("--skip-export",action="store_true", dest="skip_export")
    args = parser.parse_args()

    banner("Lower-Limb CTR-GCN — Production Training Pipeline")

    try:
        repo_info   = step1_validate_repo()
        val_info    = step2_validate_tensors(repo_info["npy_files"])
        stats       = step3_build_stats_and_splits(repo_info["npy_files"], val_info)
        cfg         = step45_setup(args)
        train_res   = step6_train(cfg)
        metrics     = step7_evaluate(train_res)
        step8_report(train_res, metrics, stats)
        if not args.skip_export:
            step9_export(train_res)
        step10_summary(train_res, metrics, stats)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user. Exiting.")
    except Exception as e:
        log.error("Pipeline failed: %s", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
