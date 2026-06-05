"""
Beta distribution policy over B-spline velocity control points.

Provides two variants:
- BetaSplinePolicy: CNN-conditioned, outputs α,β from seismic encoder.
- LearnableBetaSplinePolicy: Directly learnable α,β parameters (no encoder).

Beta(α, β) is naturally bounded on [0, 1], avoiding the gradient-vanishing
issues of Gaussian+sigmoid near boundaries. The distribution shape is flexible:
unimodal, U-shaped, J-shaped — depending on (α, β).

Reference: Chou et al. (2017), "Improving Stochastic Policy Gradients in
Continuous Control with Deep Reinforcement Learning using the Beta Distribution."
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta as TorchBeta

from .cnn_encoder import SeismicCNNEncoder


# ---------------------------------------------------------------------------
#  Utility: velocity ↔ raw bounded [0,1]
# ---------------------------------------------------------------------------

def velocity_to_unit(v: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    """Map velocity [v_min, v_max] → unit interval [0, 1]."""
    return (v - float(v_min)) / max(float(v_max) - float(v_min), 1e-12)


def unit_to_velocity(u: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    """Map unit interval [0, 1] → velocity [v_min, v_max]."""
    return float(v_min) + (float(v_max) - float(v_min)) * u


# ---------------------------------------------------------------------------
#  Beta distribution helpers
# ---------------------------------------------------------------------------

def _ensure_positive(x: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    """Ensure Beta parameters are strictly positive via softplus + offset."""
    return F.softplus(x) + float(min_val)


def beta_log_prob(
    u: torch.Tensor,
    alpha: torch.Tensor,
    beta_param: torch.Tensor,
) -> torch.Tensor:
    """Elementwise Beta log-probability for u in (0, 1).

    Args:
        u:      [..., H, W] samples in (0, 1).
        alpha:  [..., H, W] Beta α parameters.
        beta_param: [..., H, W] Beta β parameters.

    Returns:
        log_prob: [..., H, W]
    """
    dist = TorchBeta(alpha, beta_param)
    # Clamp u away from 0/1 to avoid NaN in log_prob
    u_safe = u.clamp(1e-7, 1.0 - 1e-7)
    return dist.log_prob(u_safe)


def beta_entropy(alpha: torch.Tensor, beta_param: torch.Tensor) -> torch.Tensor:
    """Entropy of Beta(α, β) distribution.

    Returns mean entropy across the control-point grid.
    """
    dist = TorchBeta(alpha, beta_param)
    return dist.entropy().mean()


# ---------------------------------------------------------------------------
#  BetaSplinePolicy: CNN → α, β
# ---------------------------------------------------------------------------

class BetaSplinePolicy(nn.Module):
    """CNN-conditioned Beta policy over B-spline velocity control points.

    Encoder: SeismicCNNEncoder (lightweight, ~50K params)
    Heads:   Two 1×1 Conv2d projecting embed_dim → 1 (α, β per control point)

    Sampling:
        u ~ Beta(α/T, β/T)  where T is the temperature
        v_ctrl = v_min + (v_max - v_min) * u
        v_model = B-spline(v_ctrl)
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        in_channels: int = 5,
        embed_dim: int = 128,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
        min_alpha: float = 0.01,
        min_beta: float = 0.01,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.min_alpha = float(min_alpha)
        self.min_beta = float(min_beta)

        self.encoder = SeismicCNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            nx_ctrl=self.nx_ctrl,
            nz_ctrl=self.nz_ctrl,
        )
        self.alpha_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.beta_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Beta parameters from seismic data.

        Args:
            x: [B, N_s, N_r, N_t] seismic shot gathers.

        Returns:
            alpha: [B, nx_ctrl, nz_ctrl]
            beta:  [B, nx_ctrl, nz_ctrl]
        """
        h = self.encoder(x)  # [B, embed_dim, nx_ctrl, nz_ctrl]
        alpha_raw = self.alpha_head(h).squeeze(1)  # [B, nx_ctrl, nz_ctrl]
        beta_raw = self.beta_head(h).squeeze(1)

        alpha = _ensure_positive(alpha_raw, self.min_alpha)
        beta_param = _ensure_positive(beta_raw, self.min_beta)

        return alpha.contiguous(), beta_param.contiguous()

    def sample(
        self,
        x: torch.Tensor,
        n: int,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Sample n velocity control grids from the Beta policy.

        Args:
            x: [B, N_s, N_r, N_t] — batch of seismic data.
            n: number of samples per batch element.
            temperature: divides α,β to control exploration (T>1 = more uniform).

        Returns:
            dict with keys:
                velocity:  [G, B, nx_ctrl, nz_ctrl] in physical m/s
                u:         [G, B, nx_ctrl, nz_ctrl] raw unit samples
                log_prob:  [G, B, nx_ctrl, nz_ctrl]
                alpha:     [B, nx_ctrl, nz_ctrl]    distribution params
                beta:      [B, nx_ctrl, nz_ctrl]
        """
        alpha, beta_param = self.forward(x)  # [B, H, W]

        # Apply temperature
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T  # [1, B, H, W]
        beta_t = beta_param.unsqueeze(0) / T

        # Sample
        dist = TorchBeta(alpha_t, beta_t)
        g = int(n)
        u = dist.sample((g,))  # [G, 1orB, B, H, W] — depends on expand
        # dist created with [1,B,H,W] params, sample((G,)) gives [G,1,B,H,W]
        # Need to fix shape handling
        u = u.squeeze(1)  # [G, B, H, W] if B>1, else handle

        # Handle B=1 case where sample may have different shape
        if u.ndim == 4 + 1:  # extra dim from TorchBeta broadcast
            u = u.squeeze(1)
        elif u.ndim == 3 and g == 1:
            u = u.unsqueeze(0)

        # Ensure correct shape
        while u.ndim > 4:
            u = u.squeeze(1)
        if u.ndim == 3:
            u = u.unsqueeze(1)  # [G, 1, H, W]

        # Compute log-prob
        log_prob = beta_log_prob(u, alpha_t.expand_as(u), beta_t.expand_as(u))

        # Map to physical velocity
        velocity = unit_to_velocity(u, self.v_min, self.v_max)

        return {
            "velocity": velocity.contiguous(),
            "u": u.contiguous(),
            "log_prob": log_prob.contiguous(),
            "alpha": alpha,
            "beta": beta_param,
        }

    def log_prob(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute log-prob of given unit samples under current policy.

        Args:
            x: [B, N_s, N_r, N_t]
            u: [G, B, nx_ctrl, nz_ctrl] unit interval samples.

        Returns:
            log_prob: [G, B, nx_ctrl, nz_ctrl]
        """
        alpha, beta_param = self.forward(x)
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        # Broadcast to match sample dims
        if alpha_t.shape[0] == 1 and u.shape[0] > 1:
            alpha_t = alpha_t.expand(u.shape[0], -1, -1, -1)
            beta_t = beta_t.expand(u.shape[0], -1, -1, -1)

        return beta_log_prob(u, alpha_t, beta_t).contiguous()


# ---------------------------------------------------------------------------
#  LearnableBetaSplinePolicy: unconditional learnable α, β
# ---------------------------------------------------------------------------

class LearnableBetaSplinePolicy(nn.Module):
    """Unconditional Beta policy — directly learnable α,β per control point.

    Useful as a case-specific baseline: no seismic encoder, pure parameter
    optimization via RL. With nx_ctrl=nz_ctrl=4, this is 32 parameters.
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
        init_alpha: float = 2.0,
        init_beta: float = 2.0,
        min_alpha: float = 0.01,
        min_beta: float = 0.01,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.min_alpha = float(min_alpha)
        self.min_beta = float(min_beta)

        # Raw parameters (unconstrained), will be mapped via softplus
        init_alpha_raw = math.log(math.exp(float(init_alpha) - float(min_alpha)) - 1.0)
        init_beta_raw = math.log(math.exp(float(init_beta) - float(min_beta)) - 1.0)

        self.alpha_raw = nn.Parameter(
            torch.full((self.nx_ctrl, self.nz_ctrl), init_alpha_raw)
        )
        self.beta_raw = nn.Parameter(
            torch.full((self.nx_ctrl, self.nz_ctrl), init_beta_raw)
        )

    def forward(self, x: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return α, β parameters (optionally broadcast to batch)."""
        alpha = _ensure_positive(self.alpha_raw, self.min_alpha)
        beta_param = _ensure_positive(self.beta_raw, self.min_beta)

        if x is not None and x.ndim >= 2:
            b = int(x.shape[0])
            alpha = alpha.unsqueeze(0).expand(b, -1, -1)
            beta_param = beta_param.unsqueeze(0).expand(b, -1, -1)
        else:
            alpha = alpha.unsqueeze(0)  # [1, H, W]
            beta_param = beta_param.unsqueeze(0)

        return alpha.contiguous(), beta_param.contiguous()

    def sample(
        self,
        x: Optional[torch.Tensor],
        n: int,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Sample n velocity control grids."""
        alpha, beta_param = self.forward(x)

        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        g = int(n)
        dist = TorchBeta(alpha_t, beta_t)
        u = dist.sample((g,))
        # Handle potential shape issues
        while u.ndim > 4:
            u = u.squeeze(1)
        if u.ndim == 3 and g > 1:
            u = u.unsqueeze(1)

        log_prob = beta_log_prob(u, alpha_t.expand_as(u), beta_t.expand_as(u))
        velocity = unit_to_velocity(u, self.v_min, self.v_max)

        return {
            "velocity": velocity.contiguous(),
            "u": u.contiguous(),
            "log_prob": log_prob.contiguous(),
            "alpha": alpha,
            "beta": beta_param,
        }

    def log_prob(
        self,
        x: Optional[torch.Tensor],
        u: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute log-prob of given unit samples."""
        alpha, beta_param = self.forward(x)
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        if alpha_t.shape[0] == 1 and u.shape[0] > 1:
            alpha_t = alpha_t.expand(u.shape[0], -1, -1, -1)
            beta_t = beta_t.expand(u.shape[0], -1, -1, -1)

        return beta_log_prob(u, alpha_t, beta_t).contiguous()

    @property
    def entropy(self) -> torch.Tensor:
        """Mean entropy across control points (detached, for logging)."""
        with torch.no_grad():
            alpha, beta_param = self.forward()
            return beta_entropy(alpha, beta_param)

    def raw_entropy(self) -> torch.Tensor:
        """Differentiable mean entropy (for entropy bonus in loss)."""
        alpha, beta_param = self.forward()
        return beta_entropy(alpha, beta_param)


# ---------------------------------------------------------------------------
#  LearnableBetaMeanPolicy: μ (mean) + κ (concentration) parameterization
# ---------------------------------------------------------------------------

class LearnableBetaMeanPolicy(nn.Module):
    """Beta policy parameterized by mean μ and concentration κ = α+β.

    Each control point has independent μ (centre of distribution) and κ
    (spread). This decouples "where" from "how narrow", enabling different
    control points to learn DIFFERENT velocity values.

    α = μ·κ,  β = (1-μ)·κ
    μ = sigmoid(μ_raw)         → [0, 1]  (target unit velocity)
    κ = softplus(κ_raw) + κ_min → [κ_min, ∞)  (concentration)
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
        init_kappa: float = 4.0,   # α+β ≈ 4 → Beta(2,2)-like spread
        kappa_min: float = 2.01,   # ensure α+β > 2 (unimodal Beta)
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.kappa_min = float(kappa_min)

        # μ_raw: random init across [0.15, 0.85] → diverse velocities
        mu_init = 0.15 + 0.70 * torch.rand(self.nx_ctrl, self.nz_ctrl)
        self.mu_raw = nn.Parameter(torch.logit(mu_init.clamp(0.01, 0.99)))

        # κ_raw: uniform concentration
        init_k_raw = math.log(
            math.exp(float(init_kappa) - float(kappa_min)) - 1.0
        )
        self.kappa_raw = nn.Parameter(
            torch.full((self.nx_ctrl, self.nz_ctrl), init_k_raw)
        )

    def _params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute μ, κ from raw parameters."""
        mu = torch.sigmoid(self.mu_raw)                      # [H, W] in (0, 1)
        kappa = F.softplus(self.kappa_raw) + self.kappa_min  # [H, W] > 2
        return mu, kappa

    def _to_alpha_beta(
        self, mu: torch.Tensor, kappa: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert μ, κ → α, β ensuring positivity."""
        alpha = mu * kappa
        beta = (1.0 - mu) * kappa
        return alpha.clamp(min=0.01), beta.clamp(min=0.01)

    def forward(
        self, x: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return α, β with optional batch broadcast."""
        mu, kappa = self._params()
        alpha, beta_param = self._to_alpha_beta(mu, kappa)

        if x is not None and x.ndim >= 2:
            b = int(x.shape[0])
            alpha = alpha.unsqueeze(0).expand(b, -1, -1)
            beta_param = beta_param.unsqueeze(0).expand(b, -1, -1)
        else:
            alpha = alpha.unsqueeze(0)
            beta_param = beta_param.unsqueeze(0)

        return alpha.contiguous(), beta_param.contiguous()

    def sample(
        self,
        x: Optional[torch.Tensor],
        n: int,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        alpha, beta_param = self.forward(x)
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        g = int(n)
        dist = TorchBeta(alpha_t, beta_t)
        u = dist.sample((g,))
        while u.ndim > 4:
            u = u.squeeze(1)
        if u.ndim == 3 and g > 1:
            u = u.unsqueeze(1)

        log_prob = beta_log_prob(u, alpha_t.expand_as(u), beta_t.expand_as(u))
        velocity = unit_to_velocity(u, self.v_min, self.v_max)

        return {
            "velocity": velocity.contiguous(),
            "u": u.contiguous(),
            "log_prob": log_prob.contiguous(),
            "alpha": alpha,
            "beta": beta_param,
        }

    def log_prob(
        self,
        x: Optional[torch.Tensor],
        u: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        alpha, beta_param = self.forward(x)
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        if alpha_t.shape[0] == 1 and u.shape[0] > 1:
            alpha_t = alpha_t.expand(u.shape[0], -1, -1, -1)
            beta_t = beta_t.expand(u.shape[0], -1, -1, -1)

        return beta_log_prob(u, alpha_t, beta_t).contiguous()

    @property
    def entropy(self) -> torch.Tensor:
        with torch.no_grad():
            alpha, beta_param = self.forward()
            return beta_entropy(alpha, beta_param)

    def raw_entropy(self) -> torch.Tensor:
        alpha, beta_param = self.forward()
        return beta_entropy(alpha, beta_param)


# ---------------------------------------------------------------------------
#  CNNBetaMeanPolicy: seismic CNN -> μ, learnable κ -> α, β
# ---------------------------------------------------------------------------

class CNNBetaMeanPolicy(nn.Module):
    """CNN-conditioned Beta policy parameterized by mean μ and concentration κ.

    This variant is designed to reuse checkpoints from ``CNNVelocityPredictor``:
    the pretrained ``pred_head`` predicts a velocity-control mean, so we load it
    directly into ``mu_head`` and then convert μ,κ to Beta α,β.
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        in_channels: int = 5,
        embed_dim: int = 128,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
        init_kappa: float = 4.0,
        kappa_min: float = 2.01,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.kappa_min = float(kappa_min)

        self.encoder = SeismicCNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            nx_ctrl=self.nx_ctrl,
            nz_ctrl=self.nz_ctrl,
        )
        self.mu_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

        init_k_raw = math.log(
            math.exp(float(init_kappa) - float(kappa_min)) - 1.0
        )
        self.kappa_raw = nn.Parameter(
            torch.full((self.nx_ctrl, self.nz_ctrl), init_k_raw)
        )

    def _to_alpha_beta(
        self, mu: torch.Tensor, kappa: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = mu * kappa
        beta = (1.0 - mu) * kappa
        return alpha.clamp(min=0.01), beta.clamp(min=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu_raw = self.mu_head(h).squeeze(1)
        mu = torch.sigmoid(mu_raw)
        kappa = F.softplus(self.kappa_raw) + self.kappa_min
        return self._to_alpha_beta(mu, kappa.unsqueeze(0))

    def sample(
        self,
        x: torch.Tensor,
        n: int,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        alpha, beta_param = self.forward(x)
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        g = int(n)
        dist = TorchBeta(alpha_t, beta_t)
        u = dist.sample((g,))
        while u.ndim > 4:
            u = u.squeeze(1)
        if u.ndim == 3:
            u = u.unsqueeze(1)

        log_prob = beta_log_prob(u, alpha_t.expand_as(u), beta_t.expand_as(u))
        velocity = unit_to_velocity(u, self.v_min, self.v_max)

        return {
            "velocity": velocity.contiguous(),
            "u": u.contiguous(),
            "log_prob": log_prob.contiguous(),
            "alpha": alpha,
            "beta": beta_param,
        }

    def log_prob(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        alpha, beta_param = self.forward(x)
        T = max(float(temperature), 1e-8)
        alpha_t = alpha.unsqueeze(0) / T
        beta_t = beta_param.unsqueeze(0) / T

        if alpha_t.shape[0] == 1 and u.shape[0] > 1:
            alpha_t = alpha_t.expand(u.shape[0], -1, -1, -1)
            beta_t = beta_t.expand(u.shape[0], -1, -1, -1)

        return beta_log_prob(u, alpha_t, beta_t).contiguous()

    def load_pretrained_velocity_predictor(self, state_dict: dict[str, torch.Tensor]):
        """Load encoder and mean head from ``CNNVelocityPredictor`` checkpoint."""
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        head_state = {
            k.removeprefix("pred_head."): v
            for k, v in state_dict.items()
            if k.startswith("pred_head.")
        }
        if not encoder_state or not head_state:
            raise ValueError(
                "Checkpoint must contain CNNVelocityPredictor keys: "
                "'encoder.*' and 'pred_head.*'"
            )
        try:
            self.encoder.load_state_dict(encoder_state)
            self.mu_head.load_state_dict(head_state)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to load pretrain checkpoint into CNNBetaMeanPolicy. "
                "Check that pretraining and RL use the same n_shots/in_channels, "
                "embed_dim, nx_ctrl, and nz_ctrl."
            ) from e


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== BetaSplinePolicy smoke test ===")
    policy_cnn = BetaSplinePolicy(nx_ctrl=4, nz_ctrl=4, in_channels=5)
    n_params = sum(p.numel() for p in policy_cnn.parameters() if p.requires_grad)
    print(f"CNN-conditioned policy: {n_params:,} params")

    dummy_seis = torch.randn(1, 5, 70, 1000)
    out = policy_cnn.sample(dummy_seis, n=8, temperature=1.0)
    print(f"  velocity shape:  {tuple(out['velocity'].shape)}")   # [8, 1, 4, 4]
    print(f"  log_prob shape:  {tuple(out['log_prob'].shape)}")
    print(f"  velocity range:  [{out['velocity'].min().item():.1f}, {out['velocity'].max().item():.1f}]")

    # Verify velocity is within bounds
    assert out["velocity"].min() >= policy_cnn.v_min - 1.0
    assert out["velocity"].max() <= policy_cnn.v_max + 1.0
    print("  ✓ velocity within physical bounds")

    print("\n=== LearnableBetaSplinePolicy smoke test ===")
    policy_learn = LearnableBetaSplinePolicy(nx_ctrl=4, nz_ctrl=4)
    n_params_l = sum(p.numel() for p in policy_learn.parameters() if p.requires_grad)
    print(f"Learnable policy: {n_params_l} params")

    out_l = policy_learn.sample(None, n=8, temperature=1.0)
    print(f"  velocity shape:  {tuple(out_l['velocity'].shape)}")
    print(f"  entropy:         {policy_learn.entropy.item():.4f}")
    print("  ✓ all checks passed")


# ---------------------------------------------------------------------------
#  GaussianMeanPolicy: Gaussian in logit space + sigmoid → bounded velocity
# ---------------------------------------------------------------------------

class GaussianMeanPolicy(nn.Module):
    """Gaussian policy over control points in logit space, squashed to [0,1] via sigmoid.

    z ~ N(μ_raw, σ²)  →  u = sigmoid(z) ∈ (0,1)  →  v = v_min + (v_max-v_min)*u

    16 independent Gaussians, one per control point.
    log_prob includes the sigmoid change-of-variables Jacobian.
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
        init_mu: float = 0.0,         # logit-space mean ≈ sigmoid(0)=0.5 → mid velocity
        init_log_sigma: float = 0.0,   # log σ → σ ≈ 1.0 in logit space
        sigma_min: float = 0.01,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.sigma_min = float(sigma_min)

        # μ: learnable logit-space mean, random init around 0
        self.mu = nn.Parameter(
            torch.randn(self.nx_ctrl, self.nz_ctrl) * 0.5 + float(init_mu)
        )
        # log σ: learnable log-std
        self.log_sigma = nn.Parameter(
            torch.full((self.nx_ctrl, self.nz_ctrl), float(init_log_sigma))
        )

    def _sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma) + self.sigma_min

    def forward(self, x: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma) with optional batch broadcast."""
        mu = self.mu
        sigma = self._sigma()
        if x is not None and x.ndim >= 2:
            b = int(x.shape[0])
            mu = mu.unsqueeze(0).expand(b, -1, -1)
            sigma = sigma.unsqueeze(0).expand(b, -1, -1)
        return mu, sigma

    def sample(
        self,
        x: Optional[torch.Tensor],
        n: int,
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        mu, sigma = self.forward(x)                                  # [1, H, W]
        T = max(float(temperature), 1e-8)
        sigma_t = sigma * T                                          # temperature scales σ

        g = int(n)
        # Sample z in logit space
        eps = torch.randn(g, 1, self.nx_ctrl, self.nz_ctrl,
                          device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(0) + sigma_t.unsqueeze(0) * eps             # [G, 1, H, W]
        u = torch.sigmoid(z)                                         # [G, 1, H, W] in (0,1)
        velocity = unit_to_velocity(u, self.v_min, self.v_max)

        # log_prob_N(z|μ,σ) and Jacobian correction
        log_prob_z = (
            -0.5 * ((z - mu.unsqueeze(0)) / sigma_t.unsqueeze(0)).pow(2)
            - torch.log(sigma_t.unsqueeze(0))
            - 0.5 * math.log(2 * math.pi)
        )  # [G, 1, H, W], per-element
        # sigmoid Jacobian: z = logit(u), du/dz = u(1-u)
        log_det = torch.log(u.clamp(1e-7, 1.0 - 1e-7) * (1.0 - u.clamp(1e-7, 1.0 - 1e-7)))
        log_prob = (log_prob_z - log_det).sum(dim=(2, 3))           # [G, 1] joint log-prob
        log_prob = log_prob.unsqueeze(-1).unsqueeze(-1)              # [G, 1, 1, 1] for PPO compat

        return {
            "velocity": velocity.contiguous(),
            "u": u.contiguous(),
            "z": z.contiguous(),
            "log_prob": log_prob.contiguous(),                       # [G, 1, 1, 1]
        }

    def log_prob(
        self,
        x: Optional[torch.Tensor],
        u: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute joint log_prob for given u ∈ (0,1) under the policy."""
        mu, sigma = self.forward(x)                                  # [1, H, W] or [B, H, W]
        T = max(float(temperature), 1e-8)
        sigma_t = sigma * T

        # Map u → z via logit
        u_clamped = u.clamp(1e-7, 1.0 - 1e-7)
        z = torch.log(u_clamped / (1.0 - u_clamped))                 # [G, B, H, W]

        if mu.shape[0] == 1 and z.shape[1] > 1:
            mu = mu.unsqueeze(0).expand(z.shape[0], -1, -1, -1)
            sigma_t = sigma_t.unsqueeze(0).expand(z.shape[0], -1, -1, -1)

        log_prob_z = (
            -0.5 * ((z - mu) / sigma_t).pow(2)
            - torch.log(sigma_t)
            - 0.5 * math.log(2 * math.pi)
        )
        log_det = torch.log(u_clamped * (1.0 - u_clamped))
        log_prob = (log_prob_z - log_det).sum(dim=(2, 3))           # [G, (B)] joint
        return log_prob.unsqueeze(-1).unsqueeze(-1).contiguous()     # [G, (B), 1, 1] for PPO compat

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the policy (per-dimension mean)."""
        with torch.no_grad():
            sigma = self._sigma()
            return 0.5 * torch.log(2.0 * math.pi * math.e * sigma.pow(2)).mean()

    def raw_entropy(self) -> torch.Tensor:
        return self.entropy
