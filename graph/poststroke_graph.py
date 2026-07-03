"""
graph/poststroke_graph.py
───────────────────────────────────────────────────────────────────────────
Custom 12-node post-stroke skeleton graph for MediaPipe Holistic pose landmarks.

Node mapping (MediaPipe landmark → graph node index):
    11 Left Shoulder  → 0     12 Right Shoulder → 1
    13 Left Elbow     → 2     14 Right Elbow    → 3
    15 Left Wrist     → 4     16 Right Wrist    → 5
    23 Left Hip       → 6     24 Right Hip      → 7
    25 Left Knee      → 8     26 Right Knee     → 9
    27 Left Ankle     → 10    28 Right Ankle    → 11

Topology:
    Left side          :  0-2 · 2-4 and 6-8 · 8-10
    Right side         :  1-3 · 3-5 and 7-9 · 9-11
    Trunk              :  0 ↔ 1 and 6 ↔ 7
    Body connections   :  0 ↔ 6 and 1 ↔ 7
    Cross stabilization:  6 ↔ 8 and 7 ↔ 9

Adjacency partitions (ST-GCN spatial partition strategy, K=3):
    A[0]  Self-links        (i → i)
    A[1]  Centripetal links (neighbour closer to root, toward Left Shoulder)
    A[2]  Centrifugal links (neighbour farther from root, away from root)

Each partition is normalised with the symmetric scheme: D^{-½} A D^{-½}.

Usage:
    python graph/poststroke_graph.py
"""

import numpy as np


class PostStrokeGraph:
    """3-partition adjacency matrix for the 12-node post-stroke skeleton."""

    NUM_NODES = 12
    ROOT_NODE = 0  # Left Shoulder as BFS root for centripetal partitioning

    SELF_LINKS = [(i, i) for i in range(NUM_NODES)]

    NEIGHBOR_LINKS = [
        # ── Left side ─────────────────────────────────────────────────────
        (0, 2), (2, 4),
        (6, 8), (8, 10),
        # ── Right side ────────────────────────────────────────────────────
        (1, 3), (3, 5),
        (7, 9), (9, 11),
        # ── Trunk ─────────────────────────────────────────────────────────
        (0, 1), (6, 7),
        # ── Body connections ──────────────────────────────────────────────
        (0, 6), (1, 7),
        # ── Cross stabilization (note: already included in left/right) ───
        (6, 8), (7, 9)
    ]

    NODE_NAMES = [
        "Left Shoulder",   "Right Shoulder",
        "Left Elbow",      "Right Elbow",
        "Left Wrist",      "Right Wrist",
        "Left Hip",        "Right Hip",
        "Left Knee",       "Right Knee",
        "Left Ankle",      "Right Ankle"
    ]

    MP_LANDMARK_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self):
        # Remove duplicate links just in case
        self.unique_links = list(set(tuple(sorted(link)) for link in self.NEIGHBOR_LINKS))
        self._hop_dist = self._bfs_distances()
        self.A = self._build_adjacency()  # shape (3, 12, 12)

    # ── BFS hop distances from ROOT_NODE ─────────────────────────────────────

    def _bfs_distances(self) -> dict[int, int]:
        adj: dict[int, list[int]] = {i: [] for i in range(self.NUM_NODES)}
        for i, j in self.unique_links:
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
        D = A.sum(axis=-1)  # (V,)
        D_safe = np.where(D > 0, D, 1.0)
        D_inv_sqrt = np.where(D > 0, D_safe ** -0.5, 0.0)  # (V,)
        return D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    # ── Build 3-partition adjacency tensor ───────────────────────────────────

    def _build_adjacency(self) -> np.ndarray:
        N = self.NUM_NODES
        A = np.zeros((3, N, N), dtype=np.float32)

        # Partition 0 – self-links
        for i, j in self.SELF_LINKS:
            A[0, i, j] = 1.0

        # Partitions 1 & 2 – centripetal / centrifugal
        for i, j in self.unique_links:
            di, dj = self._hop_dist[i], self._hop_dist[j]

            if di == dj:
                # Same depth — treat symmetrically as centripetal
                A[1, i, j] = 1.0
                A[1, j, i] = 1.0
            elif di > dj:
                # i is farther from root
                A[1, i, j] = 1.0  # i centripetally connects to j
                A[2, j, i] = 1.0  # j centrifugally connects to i
            else:
                # j is farther from root
                A[2, i, j] = 1.0  # i centrifugally connects to j
                A[1, j, i] = 1.0  # j centripetally connects to i

        # Normalise each partition independently
        for k in range(3):
            if A[k].sum() > 0:
                A[k] = self._normalize(A[k])

        return A

    # ── Pretty-print ─────────────────────────────────────────────────────────

    def print_info(self) -> None:
        bar = "=" * 58
        print(f"\n{bar}")
        print("  Post-Stroke Skeleton Graph")
        print(bar)
        print(f"  Nodes       : {self.NUM_NODES}")
        print(f"  Root node   : {self.ROOT_NODE}  ({self.NODE_NAMES[self.ROOT_NODE]})")
        print(f"  A shape     : {self.A.shape}  (K=3 partitions × V × V)\n")

        print("  Node mapping (MediaPipe landmark → graph index):")
        for idx, (mp, name) in enumerate(zip(self.MP_LANDMARK_IDS, self.NODE_NAMES)):
            print(f"    Node {idx:2d}  ←  MP-{mp:2d}  {name}")

        print("\n  Neighbour edges:")
        for i, j in sorted(self.unique_links):
            di, dj = self._hop_dist[i], self._hop_dist[j]
            print(
                f"    {self.NODE_NAMES[i]:15s} ({i}, d={di})"
                f"  ↔  {self.NODE_NAMES[j]:15s} ({j}, d={dj})"
            )

        print("\n  Adjacency partitions (non-zero entries after normalisation):")
        labels = ["Self       ", "Centripetal", "Centrifugal"]
        for k in range(3):
            nnz = int((self.A[k] > 0).sum())
            print(f"    A[{k}]  {labels[k]}  :  {nnz:3d} entries")
        print(f"{bar}\n")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    g = PostStrokeGraph()
    g.print_info()

    print("Sample A[0] (self-links, full 12×12):")
    print(np.round(g.A[0], 4))
    print("\nSample A[1] (centripetal, full 12×12):")
    print(np.round(g.A[1], 4))
    print("\nSample A[2] (centrifugal, full 12×12):")
    print(np.round(g.A[2], 4))
