import torch
import torch.nn as nn
from app.config import INPUT_SIZE, N_FRAMES, NUM_CLASSES


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=pad
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=pad
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self._pad = pad
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def _causal_trim(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        return x[:, :, :-pad] if pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self._causal_trim(out, self._pad)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self._causal_trim(out, self._pad)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        return self.act(out + res)


class SignTCN(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        channels: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 4,
        dropout: float = 0.2,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.tcn_blocks = nn.ModuleList(
            [
                TCNBlock(
                    in_ch=channels,
                    out_ch=channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SignTCN()
    x = torch.randn(8, N_FRAMES, INPUT_SIZE)
    out = model(x)
    print(f"[TCN] Input: {x.shape} -> Output: {out.shape}")
    print(f"[TCN] Parameters: {model.count_params():,}")
