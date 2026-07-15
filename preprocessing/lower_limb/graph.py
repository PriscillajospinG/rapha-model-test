"""
graph/lower_limb.py
───────────────────────────────────────────────────────────────────────────
Custom 10-node lower-limb skeleton graph for MediaPipe Pose landmarks 23-32.

Node mapping (MediaPipe landmark → graph node index):
    23 Left Hip        → 0     24 Right Hip       → 1
    25 Left Knee       → 2     26 Right Knee       → 3
    27 Left Ankle      → 4     28 Right Ankle      → 5
    29 Left Heel       → 6     30 Right Heel       → 7
    31 Left Foot Index → 8     32 Right Foot Index → 9

Topology:
    Pelvis     :  0 ↔ 1
    Left  leg  :  0-2 · 2-4 · 4-6 · 6-8
    Right leg  :  1-3 · 3-5 · 5-7 · 7-9

Adjacency partitions (ST-GCN spatial partition strategy, K=3):
    A[0]  Self-links        (i → i)
    A[1]  Centripetal links (neighbour closer to root, toward Left Hip)
    A[2]  Centrifugal links (neighbour farther from root, away from Left Hip)

Each partition is normalised with the symmetric scheme: D^{-½} A D^{-½}.
"""

import numpy as np


class LowerLimbGraph:
    """3-partition adjacency matrix for the 10-node lower-limb skeleton."""

    NUM_NODES = 10
    ROOT_NODE = 0           # Left Hip as BFS root for centripetal partitioning

    SELF_LINKS = [(i, i) for i in range(NUM_NODES)]

    NEIGHBOR_LINKS = [
        # ── Pelvis ────────────────────────────────────────────────────────
        (0, 1),
        # ── Left leg chain ────────────────────────────────────────────────
        (0, 2), (2, 4), (4, 6), (6, 8),
        # ── Right leg chain ───────────────────────────────────────────────
        (1, 3), (3, 5), (5, 7), (7, 9),
    ]

    NODE_NAMES = [
        "Left Hip",    "Right Hip",
        "Left Knee",   "Right Knee",
        "Left Ankle",  "Right Ankle",
        "Left Heel",   "Right Heel",
        "Left Foot",   "Right Foot",
    ]

    MP_LANDMARK_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self):
        self._hop_dist = self._bfs_distances()
        self.A = self._build_adjacency()   # shape (3, 10, 10)

    # ── BFS hop distances from ROOT_NODE ─────────────────────────────────────

    def _bfs_distances(self) -> dict[int, int]:
        adj: dict[int, list[int]] = {i: [] for i in range(self.NUM_NODES)}
        for i, j in self.NEIGHBOR_LINKS:
            adj[i].append(j)
            adj[j].append(i)

        dist = {i: float("inf") for i in range(self.NUM_NODES)}
        dist[self.ROOT_NODE] = 0
        queue = [self.ROOT_NODE]
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if dist[nb] == float("inf"):
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    # ── Symmetric normalisation D^{-½} A D^{-½} ─────────────────────────────

    @staticmethod
    def _normalize(A: np.ndarray) -> np.ndarray:
        D = A.sum(axis=-1)                                   # (V,)
        # Safely invert sqrt: replace zeros to avoid divide-by-zero
        D_safe      = np.where(D > 0, D, 1.0)
        D_inv_sqrt  = np.where(D > 0, D_safe ** -0.5, 0.0)   # (V,)
        return D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    # ── Build 3-partition adjacency tensor ───────────────────────────────────

    def _build_adjacency(self) -> np.ndarray:
        N = self.NUM_NODES
        A = np.zeros((3, N, N), dtype=np.float32)

        # Partition 0 – self-links
        for i, j in self.SELF_LINKS:
            A[0, i, j] = 1.0

        # Partitions 1 & 2 – centripetal / centrifugal
        for i, j in self.NEIGHBOR_LINKS:
            di, dj = self._hop_dist[i], self._hop_dist[j]

            if di == dj:
                # Same depth (e.g. the 0-1 pelvis edge when di==dj==0)
                # Treat symmetrically as centripetal on both sides
                A[1, i, j] = 1.0
                A[1, j, i] = 1.0
            elif di > dj:
                # i is farther from root → j is the centripetal direction for i
                A[1, i, j] = 1.0   # i centripetally connects to j
                A[2, j, i] = 1.0   # j centrifugally connects to i
            else:
                # j is farther from root
                A[2, i, j] = 1.0   # i centrifugally connects to j
                A[1, j, i] = 1.0   # j centripetally connects to i

        # Normalise each partition independently
        for k in range(3):
            if A[k].sum() > 0:
                A[k] = self._normalize(A[k])

        return A

    # ── Pretty-print ─────────────────────────────────────────────────────────

    def print_info(self) -> None:
        bar = "=" * 58
        print(f"\n{bar}")
        print("  Lower-Limb Skeleton Graph")
        print(bar)
        print(f"  Nodes       : {self.NUM_NODES}")
        print(f"  Root node   : {self.ROOT_NODE}  ({self.NODE_NAMES[self.ROOT_NODE]})")
        print(f"  A shape     : {self.A.shape}  (K=3 partitions × V × V)\n")

        print("  Node mapping (MediaPipe landmark → graph index):")
        for idx, (mp, name) in enumerate(zip(self.MP_LANDMARK_IDS, self.NODE_NAMES)):
            print(f"    Node {idx:2d}  ←  MP-{mp:2d}  {name}")

        print("\n  Neighbour edges:")
        for i, j in self.NEIGHBOR_LINKS:
            di, dj = self._hop_dist[i], self._hop_dist[j]
            print(
                f"    {self.NODE_NAMES[i]:12s} ({i}, d={di})"
                f"  ↔  {self.NODE_NAMES[j]:12s} ({j}, d={dj})"
            )

        print("\n  Adjacency partitions (non-zero entries after normalisation):")
        labels = ["Self       ", "Centripetal", "Centrifugal"]
        for k in range(3):
            nnz = int((self.A[k] > 0).sum())
            print(f"    A[{k}]  {labels[k]}  :  {nnz:3d} entries")
        print(f"{bar}\n")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    g = LowerLimbGraph()
    g.print_info()

    print("Sample A[0] (self-links, first 5×5):")
    print(np.round(g.A[0, :5, :5], 4))
    print("\nSample A[1] (centripetal, first 5×5):")
    print(np.round(g.A[1, :5, :5], 4))
    print("\nSample A[2] (centrifugal, first 5×5):")
    print(np.round(g.A[2, :5, :5], 4))
