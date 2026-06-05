"""
Latent-space policy for Phase III VAE-guided RL.

Uses a learnable Gaussian distribution over the VAE latent space z ∈ R^64.
The VAE decoder (frozen) maps z → velocity model → forward simulation → reward.

Two policy variants:
  - LearnableLatentPolicy: 128 params (64 μ + 64 log σ), no seismic conditioning
  - CNNLatentPolicy: CNN(seismic) → μ, σ params, same sampling
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Learnable latent mean policy (case-specific, no seismic encoder)
# ---------------------------------------------------------------------------


class LearnableLatentPolicy(nn.Module):
    """
    Learnable Gaussian in latent space.

    Parameters: 2 × latent_dim (μ + log σ for each dimension).
    z ~ N(μ, σ²), where σ = exp(log_sigma).

    Usage:
        policy = LearnableLatentPolicy(latent_dim=64)
        z_samples = policy.sample(group_size)  # [G, latent_dim]
        log_probs = policy.log_prob(z_samples)  # [G, latent_dim]
    """

    def __init__(self, latent_dim: int = 64, init_sigma: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma = nn.Parameter(torch.full((latent_dim,), math.log(init_sigma)))

    def get_dist_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma) as 1D tensors."""
        sigma = torch.exp(self.log_sigma).clamp(min=1e-4, max=10.0)
        return self.mu, sigma

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n points from the current Gaussian.

        Returns [n, latent_dim].
        """
        mu, sigma = self.get_dist_params()
        eps = torch.randn(n, self.latent_dim, device=mu.device)
        z = mu.unsqueeze(0) + eps * sigma.unsqueeze(0)
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute per-dimension log probability under current Gaussian.

        z: [G, latent_dim] → returns [G, latent_dim].
        """
        mu, sigma = self.get_dist_params()
        var = sigma.pow(2)
        log_prob = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma).unsqueeze(0)
            - 0.5 * (z - mu.unsqueeze(0)).pow(2) / var.unsqueeze(0)
        )
        return log_prob

    def entropy(self) -> torch.Tensor:
        """Differential entropy of each dimension: 0.5 * log(2πe σ²)."""
        _, sigma = self.get_dist_params()
        return 0.5 * (1.0 + math.log(2 * math.pi)) + torch.log(sigma)


# ---------------------------------------------------------------------------
# CNN-conditioned latent policy
# ---------------------------------------------------------------------------


class CNNLatentPolicy(nn.Module):
    """
    CNN encoder maps seismic data → latent distribution parameters.

    seismic [B, n_shots, n_recv, n_t] → CNN → μ [B, latent_dim], logσ [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int = 64,
        in_channels: int = 5,
        embed_dim: int = 128,
        init_sigma: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Lightweight CNN encoder (same architecture as SeismicCNNEncoder but
        # outputs to flat latent params instead of control grid)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # Final: embed_dim → latent_dim (μ) and latent_dim (logσ)
        self.head_mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, latent_dim),
        )
        self.head_logsigma = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, latent_dim),
        )

        # Initialize log_sigma bias for initial exploration
        nn.init.constant_(self.head_logsigma[1].bias, math.log(init_sigma))

    def forward(self, seismic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        seismic: [B, n_shots, n_recv, n_t] or [B, n_shots, n_t, n_recv]

        Returns (mu, log_sigma) each [B, latent_dim].
        """
        # Normalize input shape: ensure [B, C, H, W] with H=n_recv
        if seismic.ndim == 4:
            pass  # already correct
        elif seismic.ndim == 3:
            seismic = seismic.unsqueeze(0)

        features = self.encoder(seismic)
        mu = self.head_mu(features)
        log_sigma = self.head_logsigma(features)
        return mu, log_sigma

    def sample(self, seismic: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample n points per seismic input.

        Returns (z [B*n, latent_dim], mu [B*n, latent_dim], sigma [B*n, latent_dim]).
        """
        mu, log_sigma = self.forward(seismic)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)

        # Expand for n samples per input
        B = mu.shape[0]
        mu_expanded = mu.unsqueeze(1).expand(B, n, self.latent_dim).reshape(B * n, self.latent_dim)
        sigma_expanded = sigma.unsqueeze(1).expand(B, n, self.latent_dim).reshape(B * n, self.latent_dim)

        eps = torch.randn(B * n, self.latent_dim, device=mu.device)
        z = mu_expanded + eps * sigma_expanded

        return z, mu_expanded, sigma_expanded

    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute per-dimension log probability.

        z, mu, sigma: [G, latent_dim] → returns [G, latent_dim].
        """
        var = sigma.pow(2)
        log_prob = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * (z - mu).pow(2) / var
        )
        return log_prob


# ---------------------------------------------------------------------------
# Helper: VAE decoding wrapper for RL loop
# ---------------------------------------------------------------------------


class VAEDecoder:
    """
    Wrapper that loads a trained VAE decoder and normalizes/denormalizes.

    Only the decoder is used; encoder is discarded. By default all params are frozen.
    Set unfreeze_last_n > 0 to unfreeze the last N ConvTranspose layers for RL fine-tuning.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", unfreeze_last_n: int = 0):
        from agents.vae import VelocityVAE

        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["cfg"]

        self.vae = VelocityVAE(
            latent_dim=cfg["latent_dim"],
            native_size=cfg.get("native_size", 70),
            internal_size=cfg.get("internal_size", 64),
        ).to(device)
        self.vae.load_state_dict(ckpt["model_state_dict"])
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N ConvTranspose layers in decoder
        self.unfrozen_params: list[nn.Parameter] = []
        if unfreeze_last_n > 0:
            import torch.nn as nn
            deconv_layers = [m for m in self.vae.decoder.deconv if isinstance(m, nn.ConvTranspose2d)]
            n_total = len(deconv_layers)
            for i in range(max(0, n_total - unfreeze_last_n), n_total):
                for p in deconv_layers[i].parameters():
                    p.requires_grad = True
                    self.unfrozen_params.append(p)
            print(f"  Decoder: unfroze last {unfreeze_last_n}/{n_total} ConvTranspose layers "
                  f"({len(self.unfrozen_params)} params)")

        self.latent_dim = cfg["latent_dim"]
        self.v_min = cfg.get("v_min", 1500.0)
        self.v_max = cfg.get("v_max", 4500.0)
        self.native_size = cfg.get("native_size", 70)
        self.device = device

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [G, latent_dim] → velocity: [G, 1, H, W] (m/s). No gradients.
        """
        return self._decode_impl(z)

    def decode_grad(self, z: torch.Tensor) -> torch.Tensor:
        """
        Like decode() but WITH gradients (for decoder fine-tuning).
        """
        return self._decode_impl(z)

    def _decode_impl(self, z: torch.Tensor) -> torch.Tensor:
        v_norm = self.vae.decoder(z.to(self.device))
        v_norm = self.vae._to_native(v_norm)
        # Clamp to [0, 1] before denormalizing to avoid out-of-range velocities
        v_norm = v_norm.clamp(0.0, 1.0)
        v_physical = v_norm * (self.v_max - self.v_min) + self.v_min
        return v_physical

    @torch.no_grad()
    def encode(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        velocity: [B, 1, H, W] (physical)
        → z: [B, latent_dim] (latent)
        """
        v_norm = (velocity - self.v_min) / (self.v_max - self.v_min)
        mu, logvar = self.vae.encode(v_norm.to(self.device))
        return mu
