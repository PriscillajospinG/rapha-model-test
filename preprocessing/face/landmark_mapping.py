"""
graph/face_landmark_mapping.py
───────────────────────────────────────────────────────────────────────────
Maps selected MediaPipe Face Mesh landmark IDs (out of 468) to the 33
sequential graph node indices (0..32) used in the CTR-GCN face pipeline.

Landmark regions:
    Left  Eyebrow  : MP 70, 63, 105, 66, 107  → nodes  0– 4
    Right Eyebrow  : MP 336,296, 334, 293, 300 → nodes  5– 9
    Eyes           : MP 33, 133, 362, 263, 159, 145, 386, 374 → nodes 10–17
    Cheeks         : MP 50, 280, 187, 411       → nodes 18–21
    Nose           : MP 1, 4, 168               → nodes 22–24
    Mouth          : MP 61, 291, 13, 14, 78, 308, 17, 0 → nodes 25–32
"""

# Ordered list of MediaPipe landmark IDs selected for the face graph.
# Position in this list = graph node index.
FACE_LANDMARK_IDS: list[int] = [
    # ── Left Eyebrow (nodes 0–4) ─────────────────────────────────────────
    70, 63, 105, 66, 107,
    # ── Right Eyebrow (nodes 5–9) ────────────────────────────────────────
    336, 296, 334, 293, 300,
    # ── Eyes (nodes 10–17) ───────────────────────────────────────────────
    33, 133,    # left eye inner / outer corners
    362, 263,   # right eye inner / outer corners
    159, 145,   # left eye upper / lower lids
    386, 374,   # right eye upper / lower lids
    # ── Cheeks (nodes 18–21) ─────────────────────────────────────────────
    50, 280,    # left / right cheekbone lateral
    187, 411,   # left / right nasolabial
    # ── Nose (nodes 22–24) ───────────────────────────────────────────────
    1, 4, 168,
    # ── Mouth (nodes 25–32) ──────────────────────────────────────────────
    61, 291,    # left / right lip corners
    13, 14,     # upper / lower lip centre
    78, 308,    # left / right inner lip margin
    17, 0,      # chin centre / forehead centre
]

# Derived: MP-ID → node index lookup
MP_ID_TO_NODE: dict[int, int] = {
    mp_id: node_idx for node_idx, mp_id in enumerate(FACE_LANDMARK_IDS)
}

NUM_FACE_NODES: int = len(FACE_LANDMARK_IDS)   # 33

# Human-readable region labels for each node
FACE_NODE_NAMES: list[str] = [
    # Left Eyebrow (0–4)
    "L.Brow_inner", "L.Brow_2", "L.Brow_arch", "L.Brow_4", "L.Brow_outer",
    # Right Eyebrow (5–9)
    "R.Brow_inner", "R.Brow_2", "R.Brow_arch", "R.Brow_4", "R.Brow_outer",
    # Eyes (10–17)
    "L.Eye_inner", "L.Eye_outer",
    "R.Eye_inner", "R.Eye_outer",
    "L.Eye_upper", "L.Eye_lower",
    "R.Eye_upper", "R.Eye_lower",
    # Cheeks (18–21)
    "L.Cheek_lat", "R.Cheek_lat",
    "L.Cheek_naso", "R.Cheek_naso",
    # Nose (22–24)
    "Nose_bridge", "Nose_tip", "Nose_base",
    # Mouth (25–32)
    "Mouth_L_corner", "Mouth_R_corner",
    "Lip_upper_ctr", "Lip_lower_ctr",
    "Lip_L_inner", "Lip_R_inner",
    "Chin_ctr", "Forehead_ctr",
]

assert len(FACE_LANDMARK_IDS) == 33, "FACE_LANDMARK_IDS must have exactly 33 entries"
assert len(FACE_NODE_NAMES)   == 33, "FACE_NODE_NAMES must have exactly 33 entries"


if __name__ == "__main__":
    print(f"Total face nodes : {NUM_FACE_NODES}")
    print(f"\nNode  MP-ID  Region")
    print("-" * 40)
    for node_idx, (mp_id, name) in enumerate(
        zip(FACE_LANDMARK_IDS, FACE_NODE_NAMES)
    ):
        print(f"  {node_idx:2d}    {mp_id:3d}   {name}")
