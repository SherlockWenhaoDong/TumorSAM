import torch
import torch.nn as nn

class TemporalSpatialAdapter(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        # x: [B, N, C]

        out, _ = self.attn(x, x, x)

        return x + out   # residual ✅
