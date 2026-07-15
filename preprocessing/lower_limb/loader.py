"""
dataset/loader.py
───────────────────────────────────────────────────────────────────────────
PyTorch Dataset and DataLoader for the lower-limb CTR-GCN pipeline.

Expected directory layout
─────────────────────────
processed_dataset/
    skeletons/
        <sample_name>.npy       shape (4, 300, 10, 1) float64
    train_labels.csv            columns: sample_name, label
    test_labels.csv             columns: sample_name, label

Label CSV format
────────────────
    sample_name,label
    knee_Hinged Knee Fl_Ex,7
    hip_Hip Bridge,4
    ...

Labels are already integer class indices (0-8); no mapping needed.

Augmentation (training only)
─────────────────────────────
• Random Gaussian noise  N(0, σ=0.01) on x/y/z channels
• Random temporal flip   (reverse T axis) with p=0.5
• Random left↔right mirror  (swap symmetric joint pairs) with p=0.5
"""

import csv
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Expected tensor shape ─────────────────────────────────────────────────────

EXPECTED_C = 4
EXPECTED_T = 300
EXPECTED_V = 10
EXPECTED_M = 1

# Dynamic num_workers default — 0 on CPU/MPS, up to 8 on CUDA
_DEFAULT_WORKERS = min(8, os.cpu_count() or 1) if __import__('torch').cuda.is_available() else 0

# ── Class index → exercise name ───────────────────────────────────────────────

CLASS_NAMES: dict[int, str] = {
    0: "quadriceps",
    1: "calf",
    2: "leg_raise",
    3: "toes",
    4: "hip",
    5: "hamstring",
    6: "heel_slide",
    7: "knee",
    8: "ankle",
}

# ── Symmetric joint pairs for left↔right mirroring ───────────────────────────
# (left_node, right_node) — swapping these flips the bilateral skeleton

MIRROR_PAIRS = [
    (0, 1),   # Left Hip   ↔ Right Hip
    (2, 3),   # Left Knee  ↔ Right Knee
    (4, 5),   # Left Ankle ↔ Right Ankle
    (6, 7),   # Left Heel  ↔ Right Heel
    (8, 9),   # Left Foot  ↔ Right Foot
]


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PhysioSkeletonDataset(Dataset):
    """
    Loads skeleton tensors and integer class labels for CTR-GCN training.

    Args:
        labels_csv    (str):  path to CSV with columns [sample_name, label].
        skeleton_dir  (str):  directory containing ``<sample_name>.npy`` files.
        augment       (bool): apply on-the-fly data augmentation (training only).
        noise_sigma   (float): std of Gaussian noise added to x/y/z channels.
    """

    def __init__(
        self,
        labels_csv:   str,
        skeleton_dir: str,
        augment:      bool  = False,
        noise_sigma:  float = 0.01,
    ):
        self.skeleton_dir = skeleton_dir
        self.augment      = augment
        self.noise_sigma  = noise_sigma

        self.samples: list[dict] = []
        self._load_csv(labels_csv)

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # IMPORTANT: .npy filenames on disk may have trailing spaces
                # (e.g. "hip_3 Best Hip  Knee Pain .npy"). The CSV sample_name
                # column matches these exactly — only lstrip() to drop any BOM
                # or leading whitespace; do NOT rstrip() or strip().
                name = (row.get("sample_name") or "").lstrip()
                raw  = (row.get("label") or "").strip()
                if not name or not raw:
                    continue
                self.samples.append({
                    "sample_name": name,
                    "label":       int(raw),
                })
        if not self.samples:
            raise ValueError(f"No valid rows found in {path}")


    # ── Standard Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row   = self.samples[idx]
        name  = row["sample_name"]
        label = row["label"]

        data = self._load_tensor(name)          # (C, T, V, M) float32

        if self.augment:
            data = self._augment(data)

        target = torch.tensor(label, dtype=torch.long)
        return data, target, name

    # ── Tensor loading with shape validation ──────────────────────────────────

    def _load_tensor(self, sample_name: str) -> torch.Tensor:
        path = os.path.join(self.skeleton_dir, f"{sample_name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Skeleton file not found: {path}")

        arr = np.load(path).astype(np.float32)

        # Accept (C, T, V, M) or legacy (T, V, C) and fix the latter
        if arr.ndim == 3 and arr.shape == (EXPECTED_T, EXPECTED_V, EXPECTED_C):
            arr = np.transpose(arr, (2, 0, 1))[:, :, :, None]   # → (C, T, V, M)

        if arr.shape != (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M):
            raise ValueError(
                f"[{sample_name}] Bad tensor shape {arr.shape}. "
                f"Expected ({EXPECTED_C}, {EXPECTED_T}, {EXPECTED_V}, {EXPECTED_M})."
            )

        return torch.from_numpy(arr)

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (C, T, V, M)  float32

        Augmentations applied:
          1. Gaussian noise on spatial channels (x,y,z) only
          2. Temporal flip with p=0.5
          3. Left-right joint mirror with p=0.5
        """
        x = x.clone()

        # 1. Gaussian noise on channels 0,1,2 (x, y, z); skip channel 3 (visibility)
        if self.noise_sigma > 0:
            noise = torch.randn_like(x[:3]) * self.noise_sigma
            x[:3] = x[:3] + noise

        # 2. Temporal flip
        if torch.rand(1).item() < 0.5:
            x = x.flip(dims=[1])   # reverse time axis T

        # 3. Left-right mirror: swap bilateral joint pairs along V axis
        if torch.rand(1).item() < 0.5:
            x = x.clone()
            for l, r in MIRROR_PAIRS:
                tmp = x[:, :, l, :].clone()
                x[:, :, l, :] = x[:, :, r, :]
                x[:, :, r, :] = tmp

        return x

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def class_distribution(self) -> dict[int, int]:
        from collections import Counter
        return dict(Counter(s["label"] for s in self.samples))

    @property
    def present_classes(self) -> list[int]:
        return sorted(self.class_distribution.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    train_csv:    str,
    test_csv:     str,
    skeleton_dir: str,
    batch_size:   int   = 8,
    num_workers:  int   = _DEFAULT_WORKERS,
) -> tuple[PhysioSkeletonDataset, PhysioSkeletonDataset, DataLoader, DataLoader]:
    """
    Build train + test Datasets and DataLoaders.

    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    train_ds = PhysioSkeletonDataset(train_csv, skeleton_dir, augment=True)
    test_ds  = PhysioSkeletonDataset(test_csv,  skeleton_dir, augment=False)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),   # avoids worker restart between epochs
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return train_ds, test_ds, train_loader, test_loader


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    train_csv    = "datasets/lower_limb/train_labels.csv"
    test_csv     = "datasets/lower_limb/test_labels.csv"
    skeleton_dir = "datasets/lower_limb/skeletons"

    train_ds, test_ds, train_loader, test_loader = build_loaders(
        train_csv, test_csv, skeleton_dir, batch_size=4
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Test  samples : {len(test_ds)}")
    print(f"Train class dist : {train_ds.class_distribution}")
    print(f"Test  class dist : {test_ds.class_distribution}")

    batch, labels, names = next(iter(train_loader))
    print(f"Batch tensor shape : {tuple(batch.shape)}")
    print(f"Batch labels       : {labels.tolist()}")
    print(f"Batch sample names : {list(names)}")
