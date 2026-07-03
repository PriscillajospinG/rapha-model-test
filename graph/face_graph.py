"""
graph/face_graph.py
───────────────────────────────────────────────────────────────────────────
Custom 33-node facial skeleton graph for MediaPipe Face Mesh landmarks.

Node mapping (MP landmark ID → graph node index):
    Stored in graph/face_landmark_mapping.py

Topology (anatomical connections using remapped node indices):
    Left  Eyebrow chain : 0–1–2–3–4
    Right Eyebrow chain : 5–6–7–8–9
    Left  eye corners   : 10–11
    Right eye corners   : 12–13
    Left  eyelids       : 14–15
    Right eyelids       : 16–17
    Cheek lateral links : 18–22, 19–23   (cheek → nose bridge)
    Cheek nasolabial    : 20–25, 21–26   (nasolabial → lip corners)
    Nose chain          : 22–23–24
    Mouth outer         : 25–27–26  (corners → upper lip)
    Mouth lower         : 25–28–26  (corners → lower lip)
    Mouth inner         : 29–30     (inner lip margins horizontal)
    Jaw / chin          : 31–32     (chin → forehead midline)
    Midface bridge      : 23–27, 23–28  (nose tip → lip centres)
    Brow–eye links      : 2–14, 7–16    (brow arch → eyelid)

Adjacency partitions (ST-GCN spatial partition strategy, K=3):
    A[0]  Self-links        (i → i)
    A[1]  Centripetal links (toward root = node 23, Nose_tip)
    A[2]  Centrifugal links (away from root)

Each partition normalised with D^{-½} A D^{-½}.
"""

import numpy as np
from graph.face_landmark_mapping import NUM_FACE_NODES, FACE_NODE_NAMES


class FaceGraph:
    """3-partition adjacency matrix for the 33-node facial skeleton."""

    NUM_NODES = NUM_FACE_NODES   # 33
    ROOT_NODE = 23               # Nose_tip — anatomical midface anchor

    SELF_LINKS = [(i, i) for i in range(NUM_NODES)]

    NEIGHBOR_LINKS = [
        # ── Left Eyebrow chain ─────────────────────────────────────────
        (0, 1), (1, 2), (2, 3), (3, 4),
        # ── Right Eyebrow chain ────────────────────────────────────────
        (5, 6), (6, 7), (7, 8), (8, 9),
        # ── Eyes (corner pairs) ────────────────────────────────────────
        (10, 11),   # left eye inner ↔ outer
        (12, 13),   # right eye inner ↔ outer
        (14, 15),   # left upper ↔ lower eyelid
        (16, 17),   # right upper ↔ lower eyelid
        # ── Brow → eyelid links ────────────────────────────────────────
        (2, 14),    # left brow arch → left upper eyelid
        (7, 16),    # right brow arch → right upper eyelid
        # ── Cheeks lateral → nose bridge ──────────────────────────────
        (18, 22),   # left cheek lateral → nose bridge
        (19, 22),   # right cheek lateral → nose bridge
        # ── Nasolabial → lip corners ───────────────────────────────────
        (20, 25),   # left nasolabial → left lip corner
        (21, 26),   # right nasolabial → right lip corner
        # ── Nose chain ────────────────────────────────────────────────
        (22, 23), (23, 24),
        # ── Nose tip → lip centres (midface link) ─────────────────────
        (23, 27),   # nose tip → upper lip centre
        (23, 28),   # nose tip → lower lip centre
        # ── Mouth outer contour ────────────────────────────────────────
        (25, 27),   # left corner → upper lip centre
        (26, 27),   # right corner → upper lip centre
        (25, 28),   # left corner → lower lip centre
        (26, 28),   # right corner → lower lip centre
        # ── Mouth inner margin ─────────────────────────────────────────
        (29, 30),   # left inner ↔ right inner lip margin
        # ── Lip inner → outer corners ──────────────────────────────────
        (29, 25),   # left inner → left corner
        (30, 26),   # right inner → right corner
        # ── Jaw / chin midline ─────────────────────────────────────────
        (31, 28),   # chin → lower lip centre
        (32, 22),   # forehead → nose bridge
    ]

    NODE_NAMES = FACE_NODE_NAMES

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self):
        self._hop_dist = self._bfs_distances()
        self.A = self._build_adjacency()   # shape (3, 33, 33)

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
        D = A.sum(axis=-1)
        D_safe = np.where(D > 0, D, 1.0)
        D_inv_sqrt = np.where(D > 0, D_safe ** -0.5, 0.0)
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
                # Same depth → treat symmetrically as centripetal
                A[1, i, j] = 1.0
                A[1, j, i] = 1.0
            elif di > dj:
                # i is farther from root
                A[1, i, j] = 1.0   # i centripetally connects to j
                A[2, j, i] = 1.0   # j centrifugally connects to i
            else:
                # j is farther from root
                A[2, i, j] = 1.0   # i centrifugally connects to j
                A[1, j, i] = 1.0   # j centripetally connects to i

        for k in range(3):
            if A[k].sum() > 0:
                A[k] = self._normalize(A[k])

        return A

    # ── Pretty-print ─────────────────────────────────────────────────────────

    def print_info(self) -> None:
        bar = "=" * 62
        print(f"\n{bar}")
        print("  Facial Rehabilitation Skeleton Graph")
        print(bar)
        print(f"  Nodes       : {self.NUM_NODES}")
        print(f"  Root node   : {self.ROOT_NODE}  ({self.NODE_NAMES[self.ROOT_NODE]})")
        print(f"  A shape     : {self.A.shape}  (K=3 partitions × V × V)\n")

        print("  Node mapping (node index → region):")
        for idx, name in enumerate(self.NODE_NAMES):
            print(f"    Node {idx:2d}  →  {name}")

        print("\n  Neighbour edges:")
        for i, j in self.NEIGHBOR_LINKS:
            di, dj = self._hop_dist[i], self._hop_dist[j]
            print(
                f"    {self.NODE_NAMES[i]:20s} ({i:2d}, d={di})"
                f"  ↔  {self.NODE_NAMES[j]:20s} ({j:2d}, d={dj})"
            )

        print("\n  Adjacency partitions (non-zero entries after normalisation):")
        labels = ["Self       ", "Centripetal", "Centrifugal"]
        for k in range(3):
            nnz = int((self.A[k] > 0).sum())
            print(f"    A[{k}]  {labels[k]}  :  {nnz:3d} entries")
        print(f"{bar}\n")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    g = FaceGraph()
    g.print_info()
    print("A[0] shape:", g.A[0].shape)
    print("A non-zero counts:", [(g.A[k] > 0).sum() for k in range(3)])
