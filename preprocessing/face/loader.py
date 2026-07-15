"""
dataset/face_loader.py
───────────────────────────────────────────────────────────────────────────
PyTorch Dataset and DataLoader for the facial rehabilitation CTR-GCN pipeline.

Expected directory layout
─────────────────────────
processed_dataset_face/
    skeletons/
        <sample_name>.npy       shape (3, 300, 33, 1) float32
    train_labels.csv            columns: sample_name, label
    test_labels.csv             columns: sample_name, label
    face_class_map.csv          columns: class_name, label

Tensor shape
────────────
    C=3   : x, y, z (no visibility channel — FaceMesh does not provide it)
    T=300 : standardised frame count
    V=33  : facial landmark nodes
    M=1   : single tracked person

Augmentation (training only)
─────────────────────────────
    • Random Gaussian noise  N(0, σ=0.01) on x/y/z channels
    • Random temporal flip   (reverse T axis) with p=0.5
    • Random left↔right face mirror (swap bilateral landmark pairs) with p=0.5
"""

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Expected tensor shape ─────────────────────────────────────────────────────

EXPECTED_C = 3
EXPECTED_T = 300
EXPECTED_V = 33   # 33 facial landmark nodes
EXPECTED_M = 1

# Dynamic num_workers default — 0 on CPU/MPS, up to 8 on CUDA
_DEFAULT_WORKERS = min(8, os.cpu_count() or 1) if __import__('torch').cuda.is_available() else 0

# ── Bilateral mirror pairs (left node ↔ right node) ──────────────────────────
# Used for left↔right face symmetry augmentation.
# Only pairs that have a true anatomical mirror are listed.

FACE_MIRROR_PAIRS = [
    # Eyebrow: left inner ↔ right inner, arch ↔ arch, outer ↔ outer
    (0, 5),   # L.Brow_inner  ↔ R.Brow_inner
    (1, 6),   # L.Brow_2      ↔ R.Brow_2
    (2, 7),   # L.Brow_arch   ↔ R.Brow_arch
    (3, 8),   # L.Brow_4      ↔ R.Brow_4
    (4, 9),   # L.Brow_outer  ↔ R.Brow_outer
    # Eyes
    (10, 12), # L.Eye_inner   ↔ R.Eye_inner
    (11, 13), # L.Eye_outer   ↔ R.Eye_outer
    (14, 16), # L.Eye_upper   ↔ R.Eye_upper
    (15, 17), # L.Eye_lower   ↔ R.Eye_lower
    # Cheeks
    (18, 19), # L.Cheek_lat   ↔ R.Cheek_lat
    (20, 21), # L.Cheek_naso  ↔ R.Cheek_naso
    # Mouth corners
    (25, 26), # Mouth_L_corner ↔ Mouth_R_corner
    (29, 30), # Lip_L_inner    ↔ Lip_R_inner
]


# ── Helper: load dynamic class map ───────────────────────────────────────────

def load_face_class_map(class_map_csv: str) -> dict[int, str]:
    """
    Load integer-label → class-name mapping from face_class_map.csv.
    Falls back to a placeholder map if the file does not exist yet.
    """
    path = Path(class_map_csv)
    if not path.exists():
        return {
            0: "Eyebrows",
            1: "Eyes",
            2: "Frown",
            3: "Lips",
            4: "Nose",
            5: "Side Chicks",
        }

    class_map: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            class_map[int(row["label"])] = row["class_name"]
    return class_map


# ── Dataset ───────────────────────────────────────────────────────────────────

class FacePhysioDataset(Dataset):
    """
    Loads facial skeleton tensors and integer class labels for CTR-GCN.

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

        data = self._load_tensor(name)   # (C, T, V, M)

        if self.augment:
            data = self._augment(data)

        return data, torch.tensor(label, dtype=torch.long), name

    # ── Tensor loading with shape validation ──────────────────────────────────

    def _load_tensor(self, sample_name: str) -> torch.Tensor:
        path = os.path.join(self.skeleton_dir, f"{sample_name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Skeleton file not found: {path}")

        arr = np.load(path).astype(np.float32)

        expected = (EXPECTED_C, EXPECTED_T, EXPECTED_V, EXPECTED_M)
        if arr.shape != expected:
            raise ValueError(
                f"[{sample_name}] Bad tensor shape {arr.shape}. "
                f"Expected {expected}."
            )

        return torch.from_numpy(arr)

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (C=3, T, V, M)  — C channels are all positional (x, y, z)

        Augmentations:
          1. Gaussian noise on all 3 channels
          2. Temporal flip with p=0.5
          3. Left-right bilateral landmark mirror with p=0.5
        """
        x = x.clone()

        # 1. Gaussian noise on x, y, z
        if self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma

        # 2. Temporal flip
        if torch.rand(1).item() < 0.5:
            x = x.flip(dims=[1])

        # 3. Bilateral face mirror
        if torch.rand(1).item() < 0.5:
            x = x.clone()
            for l, r in FACE_MIRROR_PAIRS:
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


# ── Factory helper ────────────────────────────────────────────────────────────

def build_face_loaders(
    train_csv:    str,
    test_csv:     str,
    skeleton_dir: str,
    batch_size:   int = 8,
    num_workers:  int = _DEFAULT_WORKERS,
) -> tuple[FacePhysioDataset, FacePhysioDataset, DataLoader, DataLoader]:
    """
    Build train + test Datasets and DataLoaders for the face pipeline.

    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    train_ds = FacePhysioDataset(train_csv, skeleton_dir, augment=True)
    test_ds  = FacePhysioDataset(test_csv,  skeleton_dir, augment=False)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin,
        persistent_workers = (num_workers > 0),
        drop_last   = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
        persistent_workers = (num_workers > 0),
        drop_last   = False,
    )
    return train_ds, test_ds, train_loader, test_loader


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    base         = Path(__file__).parent.parent / "datasets/face"
    train_csv    = str(base / "train_labels.csv")
    test_csv     = str(base / "test_labels.csv")
    skeleton_dir = str(base / "skeletons")

    train_ds, test_ds, train_loader, test_loader = build_face_loaders(
        train_csv, test_csv, skeleton_dir, batch_size=4
    )

    print(f"Train samples      : {len(train_ds)}")
    print(f"Test  samples      : {len(test_ds)}")
    print(f"Train class dist   : {train_ds.class_distribution}")
    print(f"Test  class dist   : {test_ds.class_distribution}")

    batch, labels, names = next(iter(train_loader))
    print(f"Batch tensor shape : {tuple(batch.shape)}")
    print(f"Batch labels       : {labels.tolist()}")
    print(f"Batch sample names : {list(names)}")
