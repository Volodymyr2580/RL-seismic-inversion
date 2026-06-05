from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicViTControlPolicy(nn.Module):
    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        num_bins: int = 100,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        in_channels: int = 5,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.num_bins = int(num_bins)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim))
        self.query_embed = nn.Parameter(torch.randn(1, 49, embed_dim))

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=depth,
            num_decoder_layers=depth,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((self.nx_ctrl, self.nz_ctrl)),
        )
        self.head = nn.Conv2d(128, self.num_bins, kernel_size=1)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"输入必须是 4D [B,5,1000,70]，得到 {tuple(x.shape)}")
        h = self.stem(x)
        b, c, h_dim, w_dim = h.shape
        src = h.flatten(2).transpose(1, 2)
        src = src + self.pos_embed
        tgt = self.query_embed.expand(b, -1, -1)
        out = self.transformer(src, tgt)
        out = out.transpose(1, 2).reshape(b, c, h_dim, w_dim)
        out = self.decoder(out)
        logits = self.head(out)
        return logits.permute(0, 2, 3, 1).contiguous()

    def sample(self, logits: torch.Tensor, n: int = 1, *, temperature: float = 1.0, eps: float = 0.0):
        if logits.ndim != 4 or logits.shape[-1] != self.num_bins:
            raise ValueError(f"logits shape 期望 [B,nx_ctrl,nz_ctrl,{self.num_bins}]，得到 {tuple(logits.shape)}")
        temp = float(temperature)
        if temp <= 0:
            raise ValueError(f"temperature 必须 > 0，得到 {temp}")
        eps_f = float(eps)
        if not (0.0 <= eps_f < 1.0):
            raise ValueError(f"eps 必须在 [0,1) 内，得到 {eps_f}")
        probs = F.softmax(logits / temp, dim=-1)
        b, nx, nz, k = probs.shape
        if eps_f > 0.0:
            probs = probs * (1.0 - eps_f) + (eps_f / float(k))
        dist = torch.distributions.Categorical(probs=probs.reshape(-1, k))
        a_flat = dist.sample((int(n),))
        log_prob_flat = dist.log_prob(a_flat)
        a = a_flat.reshape(int(n), b, nx, nz)
        log_prob = log_prob_flat.reshape(int(n), b, nx, nz)
        return a, log_prob

