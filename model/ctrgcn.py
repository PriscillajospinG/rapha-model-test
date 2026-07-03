"""
model/ctrgcn.py
───────────────────────────────────────────────────────────────────────────
Real CTR-GCN (Channel-Topology Refinement Graph Convolutional Network)
implementation — self-contained, no external repo dependency.

Reference:
    Chen et al., "Channel-wise Topology Refinement Graph Convolution
    for Skeleton-Based Action Recognition", ICCV 2021.

Architecture for 10-joint lower-limb skeleton, 9 exercise classes:

    Data BN  (M·V·C channels)
    ↓
    Layer 1   CTR-GC + MS-TCN   in=4   → 64   stride=1  (no residual)
    Layer 2   CTR-GC + MS-TCN   64  → 64   stride=1
    Layer 3   CTR-GC + MS-TCN   64  → 64   stride=1
    Layer 4   CTR-GC + MS-TCN   64  → 128  stride=2   T: 300→150
    Layer 5   CTR-GC + MS-TCN   128 → 128  stride=1
    Layer 6   CTR-GC + MS-TCN   128 → 128  stride=1
    Layer 7   CTR-GC + MS-TCN   128 → 256  stride=2   T: 150→75
    Layer 8   CTR-GC + MS-TCN   256 → 256  stride=1
    Layer 9   CTR-GC + MS-TCN   256 → 256  stride=1
    ↓
    Global Average Pool  →  (N, 256)
    Dropout(0.5)
    Linear(256, 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Weight initialisation helpers ─────────────────────────────────────────────

def _conv_init(m: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(m.weight, mode="fan_out")
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _bn_init(m: nn.BatchNorm2d, scale: float = 1.0) -> None:
    nn.init.constant_(m.weight, scale)
    nn.init.constant_(m.bias, 0.0)


def _bn1d_init(m: nn.BatchNorm1d, scale: float = 1.0) -> None:
    nn.init.constant_(m.weight, scale)
    nn.init.constant_(m.bias, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  CTRGC  –  Channel-Topology Refinement Graph Convolution
# ─────────────────────────────────────────────────────────────────────────────

class CTRGC(nn.Module):
    """
    Channel-Topology Refinement Graph Convolution.

    For each of K=3 adjacency partitions, the effective adjacency is:

        Â_k(x) = A_k  +  α · ΔA(x)

    where the dynamic correction ΔA is computed from the input features:

        ΔA[n, i, j] = tanh(θ₁(x̄)[n,:,i])ᵀ · tanh(θ₂(x̄)[n,:,j]) / rel_c
        x̄ = temporal-mean of x

    All K aggregated feature maps are concatenated and projected with a
    shared 1×1 convolution W to produce the output.

    Args:
        in_channels  (int): number of input feature channels C.
        out_channels (int): number of output feature channels C'.
        rel_reduction (int): reduction factor for the dynamic-adjacency
            bottleneck dimension (rel_c = max(C // rel_reduction, 8)).
    """

    def __init__(self, in_channels: int, out_channels: int, rel_reduction: int = 8):
        super().__init__()
        rel_c = max(in_channels // rel_reduction, 8)

        # Dynamic adjacency branches  θ₁ and θ₂
        self.theta1 = nn.Conv2d(in_channels, rel_c, kernel_size=1, bias=False)
        self.theta2 = nn.Conv2d(in_channels, rel_c, kernel_size=1, bias=False)
        self.tanh   = nn.Tanh()

        # Learnable blending scalar α (initialised to 0 → pure static A at start)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Feature transform: concatenated K partitions → out_channels
        self.W  = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Init
        _conv_init(self.theta1)
        _conv_init(self.theta2)
        _conv_init(self.W)
        _bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, C, T, V)   input feature tensor
            A : (K, V, V)       static adjacency, K=3 partitions

        Returns:
            (N, C', T, V)
        """
        N, C, T, V = x.shape
        K = A.shape[0]           # 3

        # ── Dynamic adjacency ΔA  ──────────────────────────────────────────
        # Temporal mean → (N, rel_c, V)
        t1 = self.tanh(self.theta1(x).mean(dim=2))   # (N, rel_c, V)
        t2 = self.tanh(self.theta2(x).mean(dim=2))   # (N, rel_c, V)
        # Outer product across joints → (N, V, V)
        dA = torch.einsum("nci,ncj->nij", t1, t2) / t1.shape[1]

        # ── Per-partition graph aggregation ───────────────────────────────
        parts: list[torch.Tensor] = []
        for k in range(K):
            # Blended adjacency: static partition + learnable fraction of dynamic
            Ak = A[k] + self.alpha * dA          # (N, V, V)  (broadcast over N)
            # Aggregate neighbour features
            # y_k[n,c,t,w] = Σ_v  x[n,c,t,v] · Ak[n,v,w]
            yk = torch.einsum("nctv,nvw->nctw", x, Ak)   # (N, C, T, V)
            parts.append(yk)

        # Concatenate K feature maps and project
        y = torch.cat(parts, dim=1)    # (N, K·C, T, V)
        y = self.bn(self.W(y))         # (N, C', T, V)
        return y


# ─────────────────────────────────────────────────────────────────────────────
#  MultiScaleTCN  –  Multi-Scale Temporal Convolutional block
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleTCN(nn.Module):
    """
    Two-branch temporal convolution with different dilation rates.

    Branch 1: kernel=9, dilation=1  →  receptive field = 9 frames
    Branch 2: kernel=9, dilation=2  →  receptive field = 17 frames

    Each branch produces out_channels//2 feature maps; they are
    concatenated along the channel axis to restore out_channels.

    Args:
        in_channels  (int): input channels.
        out_channels (int): output channels (must be even).
        kernel_size  (int): temporal kernel size (default 9).
        stride       (int): temporal stride (1 or 2).
        dropout      (float): dropout applied to the output.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int  = 9,
        stride:       int  = 1,
        dropout:      float = 0.0,
    ):
        super().__init__()
        c1 = out_channels // 2
        c2 = out_channels - c1

        # Shared padding formula: pad = dilation * (kernel_size - 1) // 2
        def _pad(d: int) -> int:
            return d * (kernel_size - 1) // 2

        # ── Branch 1  dilation=1 ──────────────────────────────────────────
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c1, c1,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(_pad(1), 0),
                dilation=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(c1),
        )

        # ── Branch 2  dilation=2 ──────────────────────────────────────────
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c2, c2,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(_pad(2), 0),
                dilation=(2, 1),
                bias=False,
            ),
            nn.BatchNorm2d(c2),
        )

        self.drop = nn.Dropout(dropout)

        # Init
        for seq in [self.branch1, self.branch2]:
            for m in seq.modules():
                if isinstance(m, nn.Conv2d):
                    _conv_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    _bn_init(m, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.cat([self.branch1(x), self.branch2(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  STGCNUnit  –  Spatial-Temporal GCN block
# ─────────────────────────────────────────────────────────────────────────────

class STGCNUnit(nn.Module):
    """
    One ST-GCN block:  CTRGC → MS-TCN → residual add → ReLU.

    The residual path is:
      • Identity  if in_channels == out_channels and stride == 1
      • 1×1 Conv + BN  otherwise (handles channel expansion & temporal downsampling)
      • Zero      if residual=False (first layer only)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        A:            torch.Tensor,    # carried for shape inference only
        stride:       int   = 1,
        residual:     bool  = True,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.gcn = CTRGC(in_channels, out_channels)
        self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride, dropout=dropout)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.res: nn.Module | None = None
        elif stride == 1 and in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for m in self.res.modules():
                if isinstance(m, nn.Conv2d):
                    _conv_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    _bn_init(m, 1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : (N, C, T, V)
        A : (K, V, V)
        """
        residual = 0 if self.res is None else self.res(x)
        return self.relu(self.tcn(self.gcn(x, A)) + residual)


# ─────────────────────────────────────────────────────────────────────────────
#  Model  –  Full CTR-GCN for lower-limb exercise recognition
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    CTR-GCN for lower-limb physiotherapy exercise recognition.

    Input tensor shape : (N, C, T, V, M)
      N  = batch size
      C  = 4  (x, y, z, visibility)
      T  = 300 frames
      V  = 10 joints
      M  = 1  person

    Output : (N, num_class)  raw logits.

    Args:
        num_class   (int): number of output classes (9).
        num_point   (int): number of skeleton joints (10).
        num_person  (int): max persons per clip (1).
        in_channels (int): input feature channels (4).
        graph       : pre-built LowerLimbGraph instance (None → auto-import).
        graph_args  (dict): ignored (kept for API compatibility).
    """

    def __init__(
        self,
        num_class:   int  = 9,
        num_point:   int  = 10,
        num_person:  int  = 1,
        in_channels: int  = 4,
        graph              = None,
        graph_args:  dict = None,
    ):
        super().__init__()

        # ── Graph ─────────────────────────────────────────────────────────
        if graph is None:
            from graph.lower_limb import LowerLimbGraph
            graph_obj = LowerLimbGraph()
        else:
            graph_obj = graph

        A = torch.tensor(graph_obj.A, dtype=torch.float32)
        self.register_buffer("A", A)      # auto-moved to device with model

        # ── Data batch-normalisation ───────────────────────────────────────
        # Normalise the raw M·V·C feature stream across the temporal axis
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        _bn1d_init(self.data_bn, 1)

        # ── ST-GCN stack ───────────────────────────────────────────────────
        dp = 0.25   # per-layer dropout (regularises the small dataset)

        self.l1 = STGCNUnit(in_channels, 64,  A, stride=1, residual=False, dropout=dp)
        self.l2 = STGCNUnit(64,  64,  A, stride=1, dropout=dp)
        self.l3 = STGCNUnit(64,  64,  A, stride=1, dropout=dp)
        self.l4 = STGCNUnit(64,  128, A, stride=2, dropout=dp)   # T: 300→150
        self.l5 = STGCNUnit(128, 128, A, stride=1, dropout=dp)
        self.l6 = STGCNUnit(128, 128, A, stride=1, dropout=dp)
        self.l7 = STGCNUnit(128, 256, A, stride=2, dropout=dp)   # T: 150→75
        self.l8 = STGCNUnit(256, 256, A, stride=1, dropout=dp)
        self.l9 = STGCNUnit(256, 256, A, stride=1, dropout=dp)

        # ── Classifier head ───────────────────────────────────────────────
        self.drop_final = nn.Dropout(0.5)
        self.fc         = nn.Linear(256, num_class)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, C, T, V, M)
        Returns : (N, num_class)  logits
        """
        N, C, T, V, M = x.shape

        # ── Data BN  (normalise M·V·C feature stream) ──────────────────
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # (N, M, V, C, T)
        x = x.view(N, M * V * C, T)                  # (N, M·V·C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)                   # (N·M, C, T, V)

        # ── ST-GCN layers ───────────────────────────────────────────────
        A = self.A                  # already on the correct device via buffer
        x = self.l1(x, A)
        x = self.l2(x, A)
        x = self.l3(x, A)
        x = self.l4(x, A)
        x = self.l5(x, A)
        x = self.l6(x, A)
        x = self.l7(x, A)
        x = self.l8(x, A)
        x = self.l9(x, A)

        # ── Global average pool + classify ──────────────────────────────
        # x: (N·M, 256, T', V)  →  (N·M, 256)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # Average over persons  (N·M, 256) → (N, M, 256) → (N, 256)
        x = x.view(N, M, -1).mean(dim=1)
        x = self.drop_final(x)
        return self.fc(x)           # (N, num_class)


# ── Quick shape-check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CTR-GCN shape verification …")
    model = Model(num_class=9, num_point=10, num_person=1, in_channels=4)
    model.eval()
    dummy = torch.zeros(2, 4, 300, 10, 1)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(out.shape)}  ← expected (2, 9)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total_params:,}")
