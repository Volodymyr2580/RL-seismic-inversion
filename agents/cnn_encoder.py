"""
Lightweight CNN encoder for seismic data → B-spline control point features.

Handles the extreme aspect ratio of seismic data (N_r × N_t ≈ 70 × 1000)
using asymmetric striding: compress the receiver dimension more aggressively
than the time dimension.

Input:  p_data [B, N_s, N_r, N_t] — shot gathers
Output: features [B, embed_dim, nx_ctrl, nz_ctrl]

Architecture: 4-stage Conv2D with asymmetric strides.
Designed for ~50K parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeismicCNNEncoder(nn.Module):
    """Asymmetric CNN that maps seismic shot gathers to control grid features.

    Spatial path (receivers):    70 → 35 → 18 → 9 → pool → 4
    Temporal path (time steps):  1000 → 500 → 250 → 125 → pool → 4

    The time dimension is preserved longer to retain propagation information.
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 128,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)

        # Stage 1: compress receiver dim, keep time
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        # Stage 2: compress both, time slightly
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        # Stage 3: compress receiver, time moderately
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        # Stage 4: final compression + pooling
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.nx_ctrl, self.nz_ctrl)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_s, N_r, N_t] seismic shot gather data.

        Returns:
            features: [B, embed_dim, nx_ctrl, nz_ctrl]
        """
        if x.ndim != 4:
            raise ValueError(
                f"SeismicCNNEncoder expects [B, N_s, N_r, N_t], got shape {tuple(x.shape)}"
            )
        h = self.stage1(x)   # [B, 32, 35, 1000]
        h = self.stage2(h)   # [B, 64, 18, 500]
        h = self.stage3(h)   # [B, 128, 9, 250]
        h = self.stage4(h)   # [B, embed, 9, 125] → pool → [B, embed, 4, 4]
        return h.contiguous()


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    encoder = SeismicCNNEncoder(in_channels=5, embed_dim=128, nx_ctrl=4, nz_ctrl=4)
    print(f"SeismicCNNEncoder parameters: {count_parameters(encoder):,}")

    dummy = torch.randn(1, 5, 70, 1000)
    with torch.no_grad():
        h1 = encoder.stage1(dummy); print(f"After stage1: {tuple(h1.shape)}")  # [1,32,35,1000]
        h2 = encoder.stage2(h1);   print(f"After stage2: {tuple(h2.shape)}")  # [1,64,18,500]
        h3 = encoder.stage3(h2);   print(f"After stage3: {tuple(h3.shape)}")  # [1,128,9,250]
        h4 = encoder.stage4[:-1](h3); print(f"Before pool: {tuple(h4.shape)}")  # [1,128,9,125]
    out = encoder(dummy)
    print(f"Output: {tuple(out.shape)}")  # [1, 128, 4, 4]
