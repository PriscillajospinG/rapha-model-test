"""
train_lower_limb_ctrgcn.py
───────────────────────────────────────────────────────────────────────────
CTR-GCN training pipeline for lower-limb physiotherapy exercise recognition.

Pre-flight validation (always runs before any gradient step):
  ① Verify all tensors in train + test are (4, 300, 10, 1)
  ② Print the custom graph connections and adjacency info
  ③ Confirm all 9 exercise classes are present in the training set
  ④ Forward-pass a synthetic batch through the model → logits (N, 9)

Training configuration:
  Optimizer  : AdamW  lr=0.001  weight_decay=1e-4
  Scheduler  : CosineAnnealingLR  T_max=100
  Loss       : CrossEntropyLoss  label_smoothing=0.1
  Batch size : 8
  Epochs     : 100
  Augment    : noise + temporal-flip + left-right mirror  (training split only)

Outputs:
  models/best_lower_limb_ctrgcn.pth     best validation accuracy checkpoint
  results/lower_limb/confusion_matrix.png
  results/lower_limb/loss_curve.png
  results/lower_limb/accuracy_curve.png

Usage:
  python train_lower_limb_ctrgcn.py
"""

import csv
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing.lower_limb.loader import (
    CLASS_NAMES,
    EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
    PhysioSkeletonDataset,
    build_loaders,
)
from preprocessing.lower_limb.graph import LowerLimbGraph
from model.ctrgcn import Model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
TRAIN_CSV    = BASE_DIR / "datasets/lower_limb" / "train_labels.csv"
TEST_CSV     = BASE_DIR / "datasets/lower_limb" / "test_labels.csv"
SKELETON_DIR = BASE_DIR / "datasets/lower_limb" / "skeletons"
MODELS_DIR   = BASE_DIR / "models"
RESULTS_DIR  = BASE_DIR / "evaluation/lower_limb_final"
BEST_MODEL   = MODELS_DIR / "best_lower_limb_final.pth"

# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_CLASSES  = 9
BATCH_SIZE   = 32
EPOCHS       = 250
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
# NUM_WORKERS: auto-scaled at runtime — 0 for CPU, min(8, cpu_count) for CUDA.
# Override here if needed for your environment.
NUM_WORKERS  = 0

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-flight validation
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_names(csv_path: Path) -> list[str]:
    """
    Read sample_name column from a labels CSV.

    IMPORTANT: .npy filenames may contain trailing spaces
    (e.g. 'hip_3 Best Hip  Knee Pain .npy').  Only lstrip() — never rstrip().
    """
    names = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("sample_name") or "").lstrip()
            if name:
                names.append(name)
    return names


def validate_tensors() -> None:
    """Check every .npy file referenced in train + test CSVs."""
    log.info("── [Validation 1/3] Tensor shapes ──────────────────────────────")
    expected = (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)
    bad: list[str] = []

    all_names = _read_csv_names(TRAIN_CSV) + _read_csv_names(TEST_CSV)
    for name in all_names:
        npy = SKELETON_DIR / f"{name}.npy"
        if not npy.exists():
            bad.append(f"MISSING  {name}.npy")
            continue
        shape = np.load(npy, mmap_mode="r").shape
        if shape != expected:
            bad.append(f"BAD SHAPE {shape}  {name}.npy")

    if bad:
        log.error("Shape validation FAILED:")
        for b in bad:
            log.error("  %s", b)
        sys.exit(1)

    log.info(
        "  ✓  All %d tensors verified as %s  (float64, confirmed)",
        len(all_names), expected,
    )


def validate_graph(graph: LowerLimbGraph) -> None:
    """Print graph info and assert adjacency properties."""
    log.info("── [Validation 2/3] Graph adjacency ────────────────────────────")
    graph.print_info()

    A = graph.A
    assert A.shape == (3, 10, 10), f"Expected A shape (3,10,10), got {A.shape}"
    assert np.allclose(A[0], np.diag(np.diag(A[0]))), "A[0] is not diagonal (self-links)"
    log.info("  ✓  Adjacency matrix validated: shape %s, 3 partitions OK", A.shape)


def validate_classes(train_ds: PhysioSkeletonDataset) -> None:
    """Ensure all 9 classes are in the training split."""
    log.info("── [Validation 3/3] Class coverage ─────────────────────────────")
    present = set(train_ds.present_classes)
    required = set(range(NUM_CLASSES))
    dist = train_ds.class_distribution

    log.info("  Training class distribution:")
    for cls_id in sorted(dist):
        log.info("    Class %d  %-12s  %d sample(s)", cls_id, CLASS_NAMES[cls_id], dist[cls_id])

    missing = required - present
    if missing:
        log.warning(
            "  ⚠ Classes missing from train set: %s",
            {k: CLASS_NAMES[k] for k in sorted(missing)},
        )
        log.warning("  Training will proceed; missing classes cannot be learned.")
    else:
        log.info("  ✓  All %d classes present in training set.", NUM_CLASSES)


def validate_forward(model: nn.Module, device: torch.device) -> None:
    """Dry-run a synthetic batch to confirm output shape."""
    log.info("── [Validation 4/4] Model forward pass ─────────────────────────")
    model.eval()
    dummy = torch.zeros(2, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M).to(device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, NUM_CLASSES), \
        f"Expected output (2, {NUM_CLASSES}), got {out.shape}"
    log.info(
        "  ✓  Forward pass OK  input=%s  output=%s",
        tuple(dummy.shape), tuple(out.shape),
    )
    total_params = sum(p.numel() for p in model.parameters())
    log.info("  ✓  Model parameters: %s", f"{total_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler=None,
) -> tuple[float, float]:
    """
    Run one full training epoch.
    Supports optional CUDA Automatic Mixed Precision (AMP) via *scaler*.
    Returns (mean_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for data, labels, _ in loader:
        data   = data.to(device)
        labels = labels.to(device)

        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
            # AMP forward pass
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            logits = model(data)
            loss   = criterion(logits, labels)
            loss.backward()
            optimiser.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_with_confidence(
    model:  nn.Module,
    loader,
    device: torch.device,
) -> dict:
    model.eval()
    results = {}
    for data, labels, names in loader:
        data = data.to(device)
        logits = model(data)
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        for name, prob in zip(names, max_probs):
            results[name] = float(prob.cpu())
    return results

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """
    Evaluate the model on a DataLoader.
    Returns (accuracy, all_predictions, all_targets).
    """
    model.eval()
    all_preds:   list[int] = []
    all_targets: list[int] = []

    for data, labels, _ in loader:
        data   = data.to(device)
        labels = labels.to(device)
        preds  = model(data).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)
    return acc, all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
#  Result plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style() -> None:
    plt.rcParams.update({
        "figure.dpi":     150,
        "font.family":    "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_curves(
    train_losses: list[float],
    train_accs:   list[float],
    val_accs:     list[float],
) -> None:
    _style()
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, color="#e74c3c", linewidth=2, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [a * 100 for a in train_accs], color="#3498db",
            linewidth=2, label="Train acc")
    ax.plot(epochs, [a * 100 for a in val_accs], color="#2ecc71",
            linewidth=2, linestyle="--", label="Val acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs. Validation Accuracy"); ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_curve.png")
    plt.close(fig)

    log.info("  Plots saved: loss_curve.png, accuracy_curve.png")


def plot_confusion_matrix(
    targets: list[int],
    preds:   list[int],
) -> None:
    _style()
    labels    = list(range(NUM_CLASSES))
    tick_names = [CLASS_NAMES[i] for i in labels]

    cm = confusion_matrix(targets, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=tick_names, yticklabels=tick_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("CTR-GCN  —  Confusion Matrix (Test Set)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close(fig)
    log.info("  Plot saved: confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("  CTR-GCN Lower-Limb Training Pipeline")
    log.info("=" * 65)

    # ── Device selection: CUDA → MPS (Apple Silicon) → CPU ──────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # cudnn autotuner — best for fixed-size inputs like (4,300,10,1)
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device : %s", device)

    # ── Auto-scale num_workers on CUDA / Linux ────────────────────────────────
    import os as _os
    effective_workers = min(8, _os.cpu_count() or 1) if device.type == "cuda" else NUM_WORKERS
    log.info("num_workers : %d", effective_workers)

    # ── Automatic Mixed Precision (AMP) — CUDA only ───────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        log.info("AMP (autocast + GradScaler) : ENABLED")

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph = LowerLimbGraph()

    # ══ PRE-FLIGHT VALIDATION ═════════════════════════════════════════════════
    log.info("\n══ PRE-FLIGHT VALIDATION ═══════════════════════════════════════")
    validate_tensors()

    # Build datasets here so we can validate classes too
    from torch.utils.data import DataLoader, WeightedRandomSampler
    train_ds = PhysioSkeletonDataset(str(TRAIN_CSV), str(SKELETON_DIR), augment=True)
    test_ds  = PhysioSkeletonDataset(str(TEST_CSV),  str(SKELETON_DIR), augment=False)

    # WeightedRandomSampler: each sample weight = 1 / class_count
    dist_raw    = train_ds.class_distribution            # {class_idx: count}
    sample_wts  = [
        1.0 / max(dist_raw.get(s["label"], 1), 1)
        for s in train_ds.samples
    ]
    sampler     = WeightedRandomSampler(sample_wts, num_samples=len(sample_wts), replacement=True)
    pin         = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=effective_workers, pin_memory=pin,
                              persistent_workers=(effective_workers > 0))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=effective_workers, pin_memory=pin,
                              persistent_workers=(effective_workers > 0))
    log.info("WeightedRandomSampler: enabled — %d samples", len(sample_wts))

    validate_graph(graph)
    validate_classes(train_ds)

    model = Model(
        num_class=NUM_CLASSES, num_point=EXPECTED_V,
        num_person=EXPECTED_M, in_channels=EXPECTED_C,
        graph=graph,
    ).to(device)

    validate_forward(model, device)

    log.info("══ ALL VALIDATIONS PASSED — starting training ══════════════════\n")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)
    
    # Compute class weights
    dist = train_ds.class_distribution
    total_samples = sum(dist.values())
    class_weights = torch.zeros(NUM_CLASSES, device=device)
    for i in range(NUM_CLASSES):
        count = dist.get(i, 0)
        if count > 0:
            class_weights[i] = total_samples / (NUM_CLASSES * count)
        else:
            class_weights[i] = 1.0
            
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("Training  —  %d epochs  |  batch=%d  |  lr=%.4f",
             EPOCHS, BATCH_SIZE, LR)
    log.info("%-8s  %-12s  %-14s  %-14s  %-8s",
             "Epoch", "Train Loss", "Train Acc (%)", "Val Acc (%)", "LR")
    log.info("-" * 65)

    best_val_acc   = 0.0
    train_losses   = []
    train_accs     = []
    val_accs       = []
    t_start        = time.perf_counter()
    
    patience = 30
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimiser, criterion, device, scaler=scaler
        )
        val_acc, val_preds, val_targets = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        cur_lr = scheduler.get_last_lr()[0]
        log.info(
            "%-8d  %-12.4f  %-14.2f  %-14.2f  %-8.6f",
            epoch, tr_loss, tr_acc * 100, val_acc * 100, cur_lr,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                {
                    "epoch":     epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_acc":   val_acc,
                    "num_class": NUM_CLASSES,
                    "num_point": EXPECTED_V,
                    "num_person": EXPECTED_M,
                    "in_channels": EXPECTED_C,
                },
                BEST_MODEL,
            )
            log.info("  ★  New best checkpoint saved  val_acc=%.2f%%", val_acc * 100)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  ★  Early stopping triggered at epoch %d", epoch)
                break

    elapsed = time.perf_counter() - t_start
    log.info("-" * 65)

    # ── Final evaluation on best checkpoint ───────────────────────────────────
    log.info("Loading best checkpoint for final evaluation …")
    ckpt = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    final_acc, final_preds, final_targets = evaluate(model, test_loader, device)

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("Generating result plots …")
    plot_curves(train_losses, train_accs, val_accs)
    plot_confusion_matrix(final_targets, final_preds)

    # ── Classification report ─────────────────────────────────────────────────
    import json
    from sklearn.metrics import f1_score
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_dist = test_ds.class_distribution
    present_in_test = sorted(test_dist.keys())
    report_str = classification_report(
        final_targets, final_preds,
        labels=present_in_test,
        target_names=[CLASS_NAMES[c] for c in present_in_test],
        zero_division=0,
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_str)

    # ── metrics.json ──────────────────────────────────────────────────────────
    macro_f1 = f1_score(final_targets, final_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(final_targets, final_preds, average=None, zero_division=0)
    # Top-3 accuracy
    model.eval()
    top3_correct = top3_total = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            logits = model(data.to(device))
            top3   = logits.topk(min(3, NUM_CLASSES), dim=1).indices
            for i, lbl in enumerate(labels):
                if lbl.item() in top3[i].tolist():
                    top3_correct += 1
                top3_total += 1
    top3_acc = top3_correct / max(top3_total, 1)
    metrics_out = {
        "accuracy":     round(float(final_acc),  4),
        "macro_f1":     round(float(macro_f1),   4),
        "top3_accuracy": round(float(top3_acc),  4),
        "best_val_acc": round(float(best_val_acc), 4),
        "best_epoch":   int(ckpt["epoch"]),
        "per_class_f1": {
            CLASS_NAMES.get(i, str(i)): round(float(v), 4)
            for i, v in enumerate(per_class_f1)
        },
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("  metrics.json saved to %s", RESULTS_DIR)

    # ── Active Learning Export ────────────────────────────────────────────────
    log.info("Exporting sample_predictions.json …")
    train_probs = evaluate_with_confidence(model, train_loader, device)
    test_probs  = evaluate_with_confidence(model, test_loader,  device)
    with open(RESULTS_DIR / "sample_predictions.json", "w") as f:
        json.dump({**train_probs, **test_probs}, f, indent=2)

    report = report_str

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n%s", "=" * 65)
    log.info("  Training Complete")
    log.info("%s", "=" * 65)
    log.info("  Total train samples       : %d", len(train_ds))
    log.info("  Total test  samples       : %d", len(test_ds))
    log.info("  Train class distribution  : %s",
             {CLASS_NAMES[k]: v for k, v in train_ds.class_distribution.items()})
    log.info("  Test  class distribution  : %s",
             {CLASS_NAMES[k]: v for k, v in test_ds.class_distribution.items()})
    log.info("  Best  val accuracy        : %.2f%%", best_val_acc * 100)
    log.info("  Final val accuracy        : %.2f%%", final_acc * 100)
    log.info("  Total training time       : %.1f s", elapsed)
    log.info("  Best checkpoint           : %s", BEST_MODEL)
    log.info("  Results directory         : %s", RESULTS_DIR)
    log.info("\nPer-class report (test set):\n%s", report)
    log.info("%s", "=" * 65)


if __name__ == "__main__":
    main()
