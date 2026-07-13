"""
training/train_face_ctrgcn.py
───────────────────────────────────────────────────────────────────────────
CTR-GCN training pipeline for facial rehabilitation exercise recognition.

Pre-flight validation (always runs before any gradient step):
  ① Verify all tensors in train + test are (3, 300, 33, 1)
  ② Print the custom face graph connections and adjacency info
  ③ Confirm all exercise classes are present in the training set
  ④ Forward-pass a synthetic batch through the model → logits (N, num_classes)
  ⑤ Verify no sample appears in both train and test sets

Training configuration:
  Optimizer  : AdamW  lr=0.001  weight_decay=1e-4
  Scheduler  : CosineAnnealingLR  T_max=100
  Loss       : CrossEntropyLoss  label_smoothing=0.1
  Batch size : 8
  Epochs     : 100
  Augment    : noise + temporal-flip + bilateral face mirror (training only)

Outputs:
  models/best_face_ctrgcn.pth              best validation accuracy checkpoint
  results/lower_limb/face/loss_curve.png
  results/lower_limb/face/accuracy_curve.png
  results/lower_limb/face/confusion_matrix.png
  results/lower_limb/face/classification_report.txt

Usage:
  python training/train_face_ctrgcn.py
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
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.face_loader import (
    EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M,
    FacePhysioDataset,
    build_face_loaders,
    load_face_class_map,
)
from graph.face_graph import FaceGraph
from model.ctrgcn import Model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
TRAIN_CSV     = BASE_DIR / "datasets/face" / "train_labels.csv"
TEST_CSV      = BASE_DIR / "datasets/face" / "test_labels.csv"
SKELETON_DIR  = BASE_DIR / "datasets/face" / "skeletons"
CLASS_MAP_CSV = BASE_DIR / "datasets/face" / "face_class_map.csv"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results/lower_limb/face"
BEST_MODEL    = MODELS_DIR / "best_face_ctrgcn.pth"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 8
EPOCHS       = 100
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
        logging.FileHandler(BASE_DIR / "face_training.log", encoding="utf-8"),
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
    names = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("sample_name") or "").lstrip()
            if name:
                names.append(name)
    return names


def validate_tensors() -> None:
    """Check every .npy file referenced in train + test CSVs."""
    log.info("── [Validation 1/5] Tensor shapes ──────────────────────────────")
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

    log.info("  ✓  All %d tensors verified as %s", len(all_names), expected)


def validate_graph(graph: FaceGraph) -> None:
    """Print graph info and assert adjacency properties."""
    log.info("── [Validation 2/5] Graph adjacency ────────────────────────────")
    graph.print_info()

    A = graph.A
    assert A.shape == (3, 33, 33), f"Expected A shape (3,33,33), got {A.shape}"
    assert np.allclose(A[0], np.diag(np.diag(A[0]))), \
        "A[0] is not diagonal (self-links)"
    log.info("  ✓  Adjacency matrix validated: shape %s, 3 partitions OK", A.shape)


def validate_classes(
    train_ds: FacePhysioDataset,
    class_names: dict[int, str],
    num_classes: int,
) -> None:
    """Ensure all expected classes appear in the training set."""
    log.info("── [Validation 3/5] Class coverage ─────────────────────────────")
    present  = set(train_ds.present_classes)
    required = set(range(num_classes))
    dist     = train_ds.class_distribution

    log.info("  Training class distribution:")
    for cls_id in sorted(dist):
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        log.info("    Class %d  %-25s  %d sample(s)", cls_id, cls_name, dist[cls_id])

    missing = required - present
    if missing:
        log.warning(
            "  ⚠ Classes missing from train set: %s",
            {k: class_names.get(k, "?") for k in sorted(missing)},
        )
        log.warning("  Training will proceed; missing classes cannot be learned.")
    else:
        log.info("  ✓  All %d classes present in training set.", num_classes)


def validate_forward(
    model: nn.Module,
    device: torch.device,
    num_classes: int,
) -> None:
    """Dry-run a synthetic batch to confirm output shape."""
    log.info("── [Validation 4/5] Model forward pass ─────────────────────────")
    model.eval()
    dummy = torch.zeros(
        2, EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M
    ).to(device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, num_classes), \
        f"Expected output (2, {num_classes}), got {out.shape}"
    log.info(
        "  ✓  Forward pass OK  input=%s  output=%s",
        tuple(dummy.shape), tuple(out.shape),
    )
    total_params = sum(p.numel() for p in model.parameters())
    log.info("  ✓  Model parameters: %s", f"{total_params:,}")


def validate_no_overlap() -> None:
    """Verify no sample name appears in both train and test."""
    log.info("── [Validation 5/5] Train/test overlap ─────────────────────────")
    train_names = set(_read_csv_names(TRAIN_CSV))
    test_names  = set(_read_csv_names(TEST_CSV))
    overlap     = train_names & test_names

    if overlap:
        log.error("  ✗  Data leakage detected! Overlapping samples: %s", overlap)
        sys.exit(1)

    log.info(
        "  ✓  No overlap: train=%d  test=%d  intersection=0",
        len(train_names), len(test_names),
    )


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
    """Run one training epoch. Supports optional CUDA AMP via *scaler*."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for data, labels, _ in loader:
        data   = data.to(device)
        labels = labels.to(device)

        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
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
def evaluate(
    model:  nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
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
        "figure.dpi":        150,
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plot_curves(
    train_losses: list[float],
    train_accs:   list[float],
    val_accs:     list[float],
) -> None:
    _style()
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, color="#e74c3c", linewidth=2, label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Face CTR-GCN — Training Loss"); ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [a * 100 for a in train_accs], color="#3498db",
            linewidth=2, label="Train acc")
    ax.plot(epochs, [a * 100 for a in val_accs], color="#2ecc71",
            linewidth=2, linestyle="--", label="Val acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Face CTR-GCN — Training vs. Validation Accuracy"); ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_curve.png")
    plt.close(fig)

    log.info("  Plots saved: loss_curve.png, accuracy_curve.png")


def plot_confusion_matrix(
    targets:     list[int],
    preds:       list[int],
    class_names: dict[int, str],
    num_classes: int,
) -> None:
    _style()
    labels     = list(range(num_classes))
    tick_names = [class_names.get(i, f"class_{i}") for i in labels]

    cm = confusion_matrix(targets, preds, labels=labels)
    fig, ax = plt.subplots(
        figsize=(max(8, num_classes + 2), max(7, num_classes + 1))
    )
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=tick_names, yticklabels=tick_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("CTR-GCN Face — Confusion Matrix (Test Set)", fontsize=14)
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
    log.info("  CTR-GCN Face Rehabilitation Training Pipeline")
    log.info("=" * 65)

    # ── Device selection: CUDA → MPS (Apple Silicon) → CPU ──────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # optimal for fixed-size tensors
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

    # ── Load dynamic class map ─────────────────────────────────────────────────
    class_names = load_face_class_map(str(CLASS_MAP_CSV))   # {int → str}
    num_classes = len(class_names)
    log.info("Number of classes : %d  →  %s", num_classes, class_names)

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph = FaceGraph()

    # ══ PRE-FLIGHT VALIDATION ═════════════════════════════════════════════════
    log.info("\n══ PRE-FLIGHT VALIDATION ═══════════════════════════════════════")
    validate_tensors()

    train_ds, test_ds, train_loader, test_loader = build_face_loaders(
        str(TRAIN_CSV), str(TEST_CSV), str(SKELETON_DIR),
        batch_size=BATCH_SIZE, num_workers=effective_workers,
    )

    validate_graph(graph)
    validate_classes(train_ds, class_names, num_classes)

    model = Model(
        num_class   = num_classes,
        num_point   = EXPECTED_V,
        num_person  = EXPECTED_M,
        in_channels = EXPECTED_C,
        graph       = graph,
    ).to(device)

    validate_forward(model, device, num_classes)
    validate_no_overlap()
    log.info("══ ALL VALIDATIONS PASSED — starting training ══════════════════\n")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(
        "Training  —  %d epochs  |  batch=%d  |  lr=%.4f  |  classes=%d",
        EPOCHS, BATCH_SIZE, LR, num_classes,
    )
    log.info(
        "%-8s  %-12s  %-14s  %-14s  %-8s",
        "Epoch", "Train Loss", "Train Acc (%)", "Val Acc (%)", "LR",
    )
    log.info("-" * 65)

    best_val_acc = 0.0
    train_losses = []
    train_accs   = []
    val_accs     = []
    t_start      = time.perf_counter()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_acc":            val_acc,
                    "num_class":          num_classes,
                    "num_point":          EXPECTED_V,
                    "num_person":         EXPECTED_M,
                    "in_channels":        EXPECTED_C,
                    "class_names":        class_names,
                },
                BEST_MODEL,
            )
            log.info("  ★  New best checkpoint saved  val_acc=%.2f%%", val_acc * 100)

    elapsed = time.perf_counter() - t_start
    log.info("-" * 65)

    # ── Final evaluation on best checkpoint ───────────────────────────────────
    log.info("Loading best checkpoint for final evaluation …")
    ckpt = torch.load(BEST_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_acc, final_preds, final_targets = evaluate(model, test_loader, device)

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("Generating result plots …")
    plot_curves(train_losses, train_accs, val_accs)
    plot_confusion_matrix(final_targets, final_preds, class_names, num_classes)

    # ── Classification report ─────────────────────────────────────────────────
    test_dist       = test_ds.class_distribution
    present_in_test = sorted(test_dist.keys())
    report = classification_report(
        final_targets, final_preds,
        labels       = present_in_test,
        target_names = [class_names.get(c, f"class_{c}") for c in present_in_test],
        zero_division = 0,
    )

    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("  Classification report saved: %s", report_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n%s", "=" * 65)
    log.info("  Face Rehabilitation Training Complete")
    log.info("%s", "=" * 65)
    log.info("  Total train samples       : %d", len(train_ds))
    log.info("  Total test  samples       : %d", len(test_ds))
    log.info(
        "  Train class distribution  : %s",
        {class_names.get(k, k): v for k, v in train_ds.class_distribution.items()},
    )
    log.info(
        "  Test  class distribution  : %s",
        {class_names.get(k, k): v for k, v in test_ds.class_distribution.items()},
    )
    log.info("  Best  val accuracy        : %.2f%%", best_val_acc * 100)
    log.info("  Final val accuracy        : %.2f%%", final_acc * 100)
    log.info("  Total training time       : %.1f s", elapsed)
    log.info("  Best checkpoint           : %s", BEST_MODEL)
    log.info("  Results directory         : %s", RESULTS_DIR)
    log.info("\nPer-class report (test set):\n%s", report)
    log.info("%s", "=" * 65)


if __name__ == "__main__":
    main()
