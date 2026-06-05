"""
RL objective functions for Phase II: GDPO + GRPO-Guard policy optimization.

Key additions over Phase I:
- GRPO-Guard ratio correction: monitor and correct implicit over-optimization
- L1/L2 split reward support (two independent data misfit components)
- Beta distribution compatible log-ratio computation

Reference:
- GDPO (Dong et al., 2025, arXiv:2601.05242)
- GRPO-Guard (arXiv:2510.22319)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
#  Group & batch utilities
# ---------------------------------------------------------------------------

def group_standardize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize across the group dimension (dim=0) for each batch element.

    Args:
        x: [G, B] tensor.

    Returns:
        Standardized tensor of same shape.
    """
    if x.ndim != 2:
        raise ValueError(f"group_standardize expects [G, B], got {tuple(x.shape)}")
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(float(eps))
    return (x - mean) / std


def batch_standardize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Global standardization across all elements.

    Args:
        x: any shape.

    Returns:
        Standardized tensor of same shape.
    """
    mean = x.mean()
    std = x.std(unbiased=False).clamp_min(float(eps))
    return (x - mean) / std


# ---------------------------------------------------------------------------
#  GDPO advantage computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    """Weights for multi-objective reward components."""
    l1: float = 1.0
    l2: float = 1.0
    si: float = 0.0
    prior: float = 0.0
    tt: float = 0.0
    fwi2: float = 0.0  # second FWI reward for multi-FWI mixing

    @classmethod
    def fwi_prior(cls, l1: float = 1.0, l2: float = 1.0, prior: float = 0.05):
        return cls(l1=l1, l2=l2, si=0.0, prior=prior)


def gdpo_advantage(
    reward_parts: dict[str, torch.Tensor],
    weights: RewardWeights,
    *,
    batch_norm: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """GDPO-style decoupled advantage: per-reward group normalize → weighted sum.

    Args:
        reward_parts: dict mapping name → [G, B] reward tensor.
        weights: per-component scalar weights.
        batch_norm: apply global batch normalization after summing.
        eps: numerical stability.

    Returns:
        advantage: [G, B]
    """
    active_components = [
        ("l1", weights.l1),
        ("l2", weights.l2),
        ("si", weights.si),
        ("prior", weights.prior),
        ("tt", weights.tt),
        ("fwi2", weights.fwi2),
    ]

    components: list[torch.Tensor] = []
    for name, w in active_components:
        w = float(w)
        if w == 0.0 or name not in reward_parts:
            continue
        r = reward_parts[name]
        if r.ndim != 2:
            raise ValueError(
                f"reward_parts[{name!r}] must be [G, B], got {tuple(r.shape)}"
            )
        a = group_standardize(r, eps=eps)
        components.append(w * a)

    if not components:
        raise ValueError("No active reward components for GDPO advantage")

    adv = torch.stack(components, dim=0).sum(dim=0)

    if batch_norm and adv.numel() > 1:
        adv = batch_standardize(adv, eps=eps)

    return adv.contiguous()


# ---------------------------------------------------------------------------
#  GRPO-Guard: ratio monitoring and correction
# ---------------------------------------------------------------------------

@dataclass
class RatioGuardState:
    """Running state for GRPO-Guard ratio monitoring."""
    ratio_mean_ema: float = 1.0
    ratio_std_ema: float = 0.1
    ema_decay: float = 0.99
    correction_enabled: bool = True
    correction_threshold: float = 0.05

    def update(self, ratio_mean: float, ratio_std: float):
        """Update EMA estimates."""
        self.ratio_mean_ema = (
            self.ema_decay * self.ratio_mean_ema
            + (1.0 - self.ema_decay) * ratio_mean
        )
        self.ratio_std_ema = (
            self.ema_decay * self.ratio_std_ema
            + (1.0 - self.ema_decay) * ratio_std
        )

    def needs_correction(self) -> bool:
        """Check if ratio mean has deviated significantly from 1.0."""
        if not self.correction_enabled:
            return False
        return abs(self.ratio_mean_ema - 1.0) > self.correction_threshold


def grpo_guard_correct_log_ratio(
    log_ratio: torch.Tensor,
    guard_state: RatioGuardState | None = None,
) -> torch.Tensor:
    """Apply GRPO-Guard centering correction to log-ratio.

    When the importance ratio mean systematically deviates from 1.0
    (implicit over-optimization), we center the log-ratio by subtracting
    its spatial mean to restore unbiased gradient flow.

    Args:
        log_ratio: [G, B, H, W] elementwise log importance ratios.
        guard_state: optional state tracker for diagnostics.

    Returns:
        corrected log_ratio of same shape.
    """
    if guard_state is not None:
        ratio = torch.exp(log_ratio)
        ratio_mean_val = float(ratio.mean().detach().cpu().item())
        ratio_std_val = float(ratio.std(unbiased=False).detach().cpu().item())
        guard_state.update(ratio_mean_val, ratio_std_val)

    corrected = log_ratio
    if guard_state is not None and guard_state.needs_correction():
        # Global centering when EMA deviates. Do not subtract the per-map
        # spatial mean unconditionally: that cancels the map-level policy
        # gradient when logp_new == logp_old.
        global_bias = torch.log(
            torch.as_tensor(
                guard_state.ratio_mean_ema,
                device=log_ratio.device,
                dtype=log_ratio.dtype,
            ).clamp(min=0.01)
        )
        corrected = corrected - global_bias

    return corrected


# ---------------------------------------------------------------------------
#  Clipped policy loss
# ---------------------------------------------------------------------------

def clipped_policy_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
    *,
    guard_state: RatioGuardState | None = None,
    token_mean: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO/GRPO clipped objective for [G, B, H, W] log-prob tensors.

    Args:
        logp_new: [G, B, H, W] log-prob under current policy.
        logp_old: [G, B, H, W] log-prob under old policy.
        advantages: [G, B] advantage values.
        epsilon_low: lower clip bound (1 - ε_low).
        epsilon_high: upper clip bound (1 + ε_high).
        guard_state: optional GRPO-Guard state for monitoring.
        token_mean: average loss across all control points (True) or joint action-level (False).

    Returns:
        loss: scalar.
        stats: dict with ratio_mean, ratio_std, clip_fraction.
    """
    if logp_new.shape != logp_old.shape:
        raise ValueError(
            f"logp shape mismatch: {tuple(logp_new.shape)} vs {tuple(logp_old.shape)}"
        )
    if logp_new.ndim != 4:
        raise ValueError(f"logp must be [G, B, H, W], got {tuple(logp_new.shape)}")
    if advantages.ndim != 2:
        raise ValueError(f"advantages must be [G, B], got {tuple(advantages.shape)}")

    g, b, h, w = logp_new.shape
    if tuple(advantages.shape) != (g, b):
        raise ValueError(
            f"advantages shape mismatch: {tuple(advantages.shape)} vs {(g, b)}"
        )

    log_ratio = logp_new - logp_old

    # GRPO-Guard correction
    log_ratio_corrected = grpo_guard_correct_log_ratio(log_ratio, guard_state)

    if token_mean:
        ratio = torch.exp(log_ratio_corrected)
        adv = advantages.view(g, b, 1, 1)
        clipped = torch.clamp(
            ratio, 1.0 - float(epsilon_low), 1.0 + float(epsilon_high)
        )
        loss = -torch.minimum(ratio * adv, clipped * adv).mean()
        ratio_for_stats = ratio
    else:
        # The PPO importance ratio is for the sampled action as a whole.
        # Since control points are conditionally independent under the Beta
        # policy, joint log-prob is the sum of per-control-point log-probs.
        log_ratio_map = log_ratio_corrected.sum(dim=(2, 3))
        ratio = torch.exp(log_ratio_map)
        clipped = torch.clamp(
            ratio, 1.0 - float(epsilon_low), 1.0 + float(epsilon_high)
        )
        loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        ratio_for_stats = ratio

    with torch.no_grad():
        if token_mean:
            ratio_original = torch.exp(log_ratio)
            ratio_final = torch.exp(log_ratio_corrected)
        else:
            ratio_original = torch.exp(log_ratio.sum(dim=(2, 3)))
            ratio_final = ratio_for_stats
        low = 1.0 - float(epsilon_low)
        high = 1.0 + float(epsilon_high)
        stats = {
            "loss": float(loss.detach().cpu().item()),
            "ratio_mean": float(ratio_original.mean().detach().cpu().item()),
            "ratio_std": float(ratio_original.std(unbiased=False).detach().cpu().item()),
            "ratio_corrected_mean": float(ratio_final.mean().detach().cpu().item()),
            "ratio_corrected_std": float(ratio_final.std(unbiased=False).detach().cpu().item()),
            "clip_fraction": float(
                ((ratio_final < low) | (ratio_final > high))
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            "guard_correction_active": (
                guard_state.needs_correction() if guard_state else False
            ),
        }

    return loss, stats


# ---------------------------------------------------------------------------
#  Reward components
# ---------------------------------------------------------------------------

def sign_preserving_log(
    x: torch.Tensor, k: float = 3.0, c: float = 0.0, eps: float = 1e-6
) -> torch.Tensor:
    """Sign-preserving log transform for seismic amplitude compression."""
    return torch.sign(x) * torch.log(k * torch.abs(x) + c + eps)


def reward_l1(
    pred_seismic: torch.Tensor,
    obs_seismic: torch.Tensor,
    k: float = 3.0,
    c: float = 0.0,
) -> torch.Tensor:
    """L1 data misfit reward (negative L1 error).

    Args:
        pred_seismic: [B, N_s, N_t, N_r] predicted seismograms.
        obs_seismic:  [B, N_s, N_t, N_r] observed seismograms.

    Returns:
        reward: [B] negative L1 error per batch element.
    """
    pred_t = sign_preserving_log(pred_seismic, k=k, c=c)
    obs_t = sign_preserving_log(obs_seismic, k=k, c=c)
    return -torch.abs(pred_t - obs_t).sum(dim=(1, 2, 3))


def reward_l2(
    pred_seismic: torch.Tensor,
    obs_seismic: torch.Tensor,
    k: float = 3.0,
    c: float = 0.0,
) -> torch.Tensor:
    """L2 data misfit reward (negative L2 error).

    Args:
        pred_seismic: [B, N_s, N_t, N_r] predicted seismograms.
        obs_seismic:  [B, N_s, N_t, N_r] observed seismograms.

    Returns:
        reward: [B] negative L2 error per batch element.
    """
    pred_t = sign_preserving_log(pred_seismic, k=k, c=c)
    obs_t = sign_preserving_log(obs_seismic, k=k, c=c)
    return -((pred_t - obs_t) ** 2).sum(dim=(1, 2, 3))


def velocity_prior_reward(
    v_model: torch.Tensor,
    *,
    v_min: float = 1500.0,
    v_max: float = 4500.0,
    smooth_weight: float = 1.0,
    monotonic_weight: float = 0.1,
    bound_weight: float = 1.0,
) -> torch.Tensor:
    """Geophysical prior reward for [B, nx, nz] velocity models.

    Penalizes: roughness, inverted velocity gradient, out-of-range values.
    """
    if v_model.ndim != 3:
        raise ValueError(f"v_model must be [B, nx, nz], got {tuple(v_model.shape)}")

    dx = v_model[:, 1:, :] - v_model[:, :-1, :]
    dz = v_model[:, :, 1:] - v_model[:, :, :-1]

    smooth_pen = (dx.pow(2).mean(dim=(1, 2)) + dz.pow(2).mean(dim=(1, 2)))
    monotonic_pen = torch.relu(-dz).mean(dim=(1, 2))
    bound_pen = (
        torch.relu(float(v_min) - v_model).pow(2).mean(dim=(1, 2))
        + torch.relu(v_model - float(v_max)).pow(2).mean(dim=(1, 2))
    )

    return -(
        float(smooth_weight) * smooth_pen
        + float(monotonic_weight) * monotonic_pen
        + float(bound_weight) * bound_pen
    )
