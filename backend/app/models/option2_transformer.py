import math
import torch
import torch.nn as nn
from app.config import INPUT_SIZE, N_FRAMES, NUM_CLASSES


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SignTransformer(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        fc_hidden: int = 64,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        assert d_model % nhead == 0

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=N_FRAMES + 1, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.classifier(cls_out)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SignTransformer()
    x = torch.randn(8, N_FRAMES, INPUT_SIZE)
    out = model(x)
    print(f"[Transformer] Input: {x.shape} -> Output: {out.shape}")
    print(f"[Transformer] Parameters: {model.count_params():,}")
