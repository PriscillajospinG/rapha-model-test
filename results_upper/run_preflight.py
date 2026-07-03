"""
results_upper/run_preflight.py
──────────────────────────────
Standalone pre-flight validator for the upper-limb pipeline.
Checks:
  1. All tensors are (4, 300, 8, 1)
  2. Graph A.shape == (3, 8, 8)
  3. Forward pass (2, 4, 300, 8, 1) → (2, num_classes)
  4. All train classes exist
  5. No overlap between train/test sets

Saves: results_upper/validation_report.txt
"""
import csv
import sys
import numpy as np
from pathlib import Path

BASE_DIR     = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

TRAIN_CSV    = BASE_DIR / "processed_dataset_upper" / "train_labels.csv"
TEST_CSV     = BASE_DIR / "processed_dataset_upper" / "test_labels.csv"
SKELETON_DIR = BASE_DIR / "processed_dataset_upper" / "skeletons"
CLASS_MAP    = BASE_DIR / "processed_dataset_upper" / "upper_class_map.csv"
RESULTS_DIR  = BASE_DIR / "results_upper"
REPORT_PATH  = RESULTS_DIR / "validation_report.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = (4, 300, 8, 1)
lines = []

def log(msg=""):
    print(msg)
    lines.append(msg)

def read_csv(p):
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("sample_name") or "").lstrip()
            lbl  = (r.get("label") or "").strip()
            if name and lbl:
                rows.append((name, int(lbl)))
    return rows

log("=" * 60)
log("  Upper-Limb Pre-Flight Validation Report")
log("=" * 60)

# ── 1. Tensor shapes ─────────────────────────────────────────────
log("\n[1/5] Tensor shape validation")
train_rows = read_csv(TRAIN_CSV)
test_rows  = read_csv(TEST_CSV)
all_rows   = train_rows + test_rows

bad = []
missing = []
for name, _ in all_rows:
    npy = SKELETON_DIR / f"{name}.npy"
    if not npy.exists():
        missing.append(name)
        continue
    shape = np.load(npy, mmap_mode="r").shape
    if shape != EXPECTED:
        bad.append(f"  BAD SHAPE {shape}  {name}.npy")

if missing:
    log(f"  [WARN] {len(missing)} missing .npy files:")
    for m in missing[:10]:
        log(f"    {m}")
if bad:
    log(f"  [FAIL] {len(bad)} wrong-shape tensors:")
    for b in bad:
        log(b)
else:
    log(f"  ✓  All {len(all_rows)} tensors have shape {EXPECTED}")

# ── 2. Graph ─────────────────────────────────────────────────────
log("\n[2/5] Graph adjacency validation")
try:
    from graph.upper_limb import UpperLimbGraph
    g = UpperLimbGraph()
    assert g.A.shape == (3, 8, 8), f"Shape mismatch: {g.A.shape}"
    assert np.allclose(g.A[0], np.diag(np.diag(g.A[0]))), "A[0] not diagonal"
    log(f"  ✓  A.shape = {g.A.shape}")
    log(f"  ✓  A[0] is diagonal (self-links)")
    log(f"  Graph connections: {g.NEIGHBOR_LINKS}")
except Exception as e:
    log(f"  [FAIL] {e}")

# ── 3. Forward pass ───────────────────────────────────────────────
log("\n[3/5] Model forward pass")
try:
    import torch
    from dataset.upper_loader import load_class_map
    from model.ctrgcn import Model
    from graph.upper_limb import UpperLimbGraph

    class_names = load_class_map(str(CLASS_MAP))
    num_classes = len(class_names)
    g = UpperLimbGraph()
    model = Model(num_class=num_classes, num_point=8, num_person=1,
                  in_channels=4, graph=g)
    model.eval()
    dummy = torch.zeros(2, 4, 300, 8, 1)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, num_classes), f"Output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  ✓  Input  : {tuple(dummy.shape)}")
    log(f"  ✓  Output : {tuple(out.shape)}  (num_classes={num_classes})")
    log(f"  ✓  Parameters : {n_params:,}")
except Exception as e:
    log(f"  [FAIL] {e}")

# ── 4. Class coverage ─────────────────────────────────────────────
log("\n[4/5] Class coverage in training set")
try:
    train_labels = set(lbl for _, lbl in train_rows)
    all_classes  = set(range(num_classes))
    missing_cls  = all_classes - train_labels
    log(f"  Train labels present: {sorted(train_labels)}")
    if missing_cls:
        log(f"  [WARN] Missing classes in train: {missing_cls}")
    else:
        log(f"  ✓  All {num_classes} classes represented in training set")
except Exception as e:
    log(f"  [FAIL] {e}")

# ── 5. Train/test overlap ─────────────────────────────────────────
log("\n[5/5] Train/test overlap check")
train_names = set(n for n, _ in train_rows)
test_names  = set(n for n, _ in test_rows)
overlap     = train_names & test_names
if overlap:
    log(f"  [FAIL] {len(overlap)} samples appear in both splits: {overlap}")
else:
    log(f"  ✓  No overlap between train ({len(train_names)}) and test ({len(test_names)}) sets")

log("\n" + "=" * 60)
log("  Validation Complete")
log("=" * 60)

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport saved: {REPORT_PATH}")

if bad or missing:
    sys.exit(1)
