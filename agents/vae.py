"""
Velocity VAE for Phase III: latent-space velocity model representation.

Architecture:
  Encoder: 70×70 → resize 64×64 → 4-stage Conv2d → FC → μ(64), logvar(64)
  Decoder: z(64) → FC → 4-stage ConvTranspose2d → 64×64 → resize 70×70

The VAE learns a 64-dimensional smooth latent manifold of velocity models.
During RL, the decoder is frozen — RL searches in latent space z ∈ R^64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityEncoder(nn.Module):
    """Conv encoder: 64×64 velocity → 64D latent (μ, logvar)."""

    def __init__(self, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        # 64 → 32 → 16 → 8 → 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8×8
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 4×4
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        # Flatten: 256 * 4 * 4 = 4096
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar) each of shape [B, latent_dim]."""
        h = self.conv(x)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VelocityDecoder(nn.Module):
    """Conv decoder: 64D latent → 64×64 velocity."""

    def __init__(self, latent_dim: int = 64, out_channels: int = 1):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.SiLU(),
        )

        # 4×4 → 8×8 → 16×16 → 32×32 → 64×64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns velocity model [B, out_channels, 64, 64]."""
        h = self.fc(z)
        h = h.view(h.shape[0], 256, 4, 4)
        x = self.deconv(h)
        return x


class VelocityVAE(nn.Module):
    """
    VAE for velocity models.

    Input/output: [B, 1, H, W] where (H, W) can be 70×70 (native) or 64×64.
    Internal processing is at 64×64 for clean Conv symmetry.

    Usage:
        vae = VelocityVAE()
        x_70 = torch.randn(8, 1, 70, 70)
        recon_70, mu, logvar = vae(x_70)
        loss = vae.loss_function(recon_70, x_70, mu, logvar, beta=1.0)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        native_size: int = 70,
        internal_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.native_size = native_size
        self.internal_size = internal_size

        self.encoder = VelocityEncoder(latent_dim=latent_dim)
        self.decoder = VelocityDecoder(latent_dim=latent_dim)

    def _to_internal(self, x: torch.Tensor) -> torch.Tensor:
        """Resize native → internal if needed."""
        if x.shape[-1] == self.internal_size and x.shape[-2] == self.internal_size:
            return x
        return F.interpolate(x, size=(self.internal_size, self.internal_size),
                            mode='bilinear', align_corners=False)

    def _to_native(self, x: torch.Tensor) -> torch.Tensor:
        """Resize internal → native if needed."""
        if x.shape[-1] == self.native_size and x.shape[-2] == self.native_size:
            return x
        return F.interpolate(x, size=(self.native_size, self.native_size),
                            mode='bilinear', align_corners=False)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, 1, H, W] → (mu, logvar) [B, latent_dim]."""
        x_int = self._to_internal(x)
        return self.encoder(x_int)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] → velocity [B, 1, native, native]."""
        x_int = self.decoder(z)
        return self._to_native(x_int)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        β-VAE loss: MSE(recon, target) + β * KL(N(μ,σ) || N(0,1)).

        MSE/MAE are per-pixel (averaged over batch × spatial).
        KL is per-sample (averaged over batch). Returns (loss, metrics_dict).
        """
        batch_size = target.shape[0]
        n_pixels = target.shape[2] * target.shape[3]  # H × W

        # Per-pixel MSE
        mse = F.mse_loss(recon, target, reduction='mean')  # mean over all elements
        # Per-sample KL
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Weight KL relative to per-pixel MSE: scale by 1/n_pixels to keep
        # β interpretable across image sizes
        loss = mse + beta * kl / n_pixels

        with torch.no_grad():
            mae = F.l1_loss(recon, target, reduction='mean')

        metrics = {
            'loss': loss.item(),
            'mse': mse.item(),
            'mae': mae.item(),
            'kl': kl.item(),
        }
        return loss, metrics

    def decode_many(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent vectors → velocity models.

        Used in RL rollout: z [G, latent_dim] → v [G, 1, native, native].
        """
        v_int = self.decoder(z)  # [G, 1, 64, 64]
        return self._to_native(v_int)
