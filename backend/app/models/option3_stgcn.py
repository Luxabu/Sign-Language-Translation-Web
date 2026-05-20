import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config import N_COORDS, N_FRAMES, N_LANDMARKS, NUM_CLASSES


POSE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
]


def _hand_edges(offset: int):
    raw = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (5, 9),
        (9, 13),
        (13, 17),
    ]
    return [(a + offset, b + offset) for a, b in raw]


ALL_EDGES = POSE_EDGES + _hand_edges(33) + _hand_edges(54)
CROSS_EDGES = [(15, 33), (16, 54)]
ALL_EDGES += CROSS_EDGES

N = N_LANDMARKS


def _build_adjacency() -> torch.Tensor:
    A = np.eye(N, dtype=np.float32)
    for i, j in ALL_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1) ** -0.5)
    A = D @ A @ D
    return torch.from_numpy(A)


class SpatialGCN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.W = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.W(x)
        return self.bn(x)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        A: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn = SpatialGCN(in_ch, out_ch, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.gcn(x))
        x = self.tcn(x) + res
        return self.relu(x)


class SignSTGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = N_COORDS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.25,
    ):
        super().__init__()
        A = _build_adjacency()
        self.data_bn = nn.BatchNorm1d(in_channels * N)
        self.layers = nn.ModuleList(
            [
                STGCNBlock(2, 64, A, dropout=dropout),
                STGCNBlock(64, 64, A, dropout=dropout),
                STGCNBlock(64, 64, A, dropout=dropout),
                STGCNBlock(64, 128, A, dropout=dropout),
                STGCNBlock(128, 128, A, dropout=dropout),
                STGCNBlock(128, 256, A, dropout=dropout),
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, N_LANDMARKS, N_COORDS)
        x = x.permute(0, 3, 1, 2)
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(B, N_COORDS * N_LANDMARKS, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(B, N_COORDS, N_LANDMARKS, T).permute(0, 1, 3, 2)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x).view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SignSTGCN()
    x = torch.randn(8, N_FRAMES, 150)
    out = model(x)
    print(f"[ST-GCN] Input: {x.shape} -> Output: {out.shape}")
    print(f"[ST-GCN] Parameters: {model.count_params():,}")
