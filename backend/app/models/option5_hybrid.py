import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config import (
    INPUT_SIZE,
    N_FRAMES,
    NUM_CLASSES,
    CNN_CHANNELS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    ATTN_HEADS,
    DROP_CNN,
    DROP_LSTM,
    DROP_CLS,
    FC_HIDDEN,
)


class MultiScaleConvModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        mid = out_ch // 3
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(mid * 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        cat = torch.cat([b3, b5, b7], dim=1)
        out = self.fuse(cat) + self.shortcut(x)
        return F.gelu(out)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.query.view(1, embed_dim, 1))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(self.query.expand(B, 1, D))
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, 1, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).squeeze(-2)
        out = out.reshape(B, D)
        out = self.out_proj(out)
        return self.norm(out)


class TemporalConvStack(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = MultiScaleConvModule(in_ch, out_ch, dropout)
        self.block2 = MultiScaleConvModule(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class SignHybrid(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        cnn_channels: int = CNN_CHANNELS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        attn_heads: int = ATTN_HEADS,
        fc_hidden: int = FC_HIDDEN,
        drop_cnn: float = DROP_CNN,
        drop_lstm: float = DROP_LSTM,
        drop_cls: float = DROP_CLS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, cnn_channels, bias=False),
            nn.LayerNorm(cnn_channels),
            nn.GELU(),
        )
        self.cnn = TemporalConvStack(
            in_ch=cnn_channels,
            out_ch=cnn_channels,
            dropout=drop_cnn,
        )
        lstm_out = lstm_hidden * 2
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_lstm if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_out)
        self.attn_pool = AttentionPooling(
            embed_dim=lstm_out,
            num_heads=attn_heads,
            dropout=drop_cnn,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.GELU(),
            nn.Dropout(drop_cls),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(drop_cls * 0.5),
            nn.Linear(fc_hidden // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.lstm_norm(x)
        x = self.attn_pool(x)
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SignHybrid()
    x = torch.randn(8, N_FRAMES, INPUT_SIZE)
    logits = model(x)
    print(f"[Hybrid] Input:   {x.shape}")
    print(f"[Hybrid] Output:  {logits.shape}")
    print(f"[Hybrid] Params:  {model.count_params():,}")
