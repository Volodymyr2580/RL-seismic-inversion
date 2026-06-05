from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def velocity_to_raw(
    velocity: torch.Tensor,
    v_min: float,
    v_max: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Map bounded velocity values to unconstrained Gaussian action space."""
    x = (velocity - float(v_min)) / max(float(v_max) - float(v_min), 1e-12)
    x = x.clamp(float(eps), 1.0 - float(eps))
    return torch.logit(x)


def raw_to_velocity(raw_action: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    """Map unconstrained Gaussian actions to bounded velocity control points."""
    return float(v_min) + (float(v_max) - float(v_min)) * torch.sigmoid(raw_action)


def gaussian_log_prob(raw_action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Elementwise Normal log probability without summing over the control grid."""
    std = torch.exp(log_std)
    var = std * std
    return -0.5 * (((raw_action - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))


class SeismicContinuousSplinePolicy(nn.Module):
    """Seismic-conditioned Gaussian policy over B-spline velocity control points.

    The policy predicts a Normal distribution in an unconstrained action space.
    Samples are mapped through a sigmoid into physical velocity bounds before
    B-spline reconstruction.
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        in_channels: int = 5,
        embed_dim: int = 256,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.nx_ctrl, self.nz_ctrl)),
        )
        self.mean_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.log_std_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"input must be [B,C,T,R], got {tuple(x.shape)}")
        h = self.encoder(x)
        mean = self.mean_head(h).squeeze(1)
        log_std = self.log_std_head(h).squeeze(1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean.contiguous(), log_std.contiguous()

    def sample(
        self,
        x: torch.Tensor,
        n: int,
        *,
        v_min: float,
        v_max: float,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.forward(x)
        log_std_s = log_std + math.log(max(float(temperature), 1e-8))
        g = int(n)
        eps = torch.randn((g,) + tuple(mean.shape), device=mean.device, dtype=mean.dtype)
        raw_action = mean.unsqueeze(0) + torch.exp(log_std_s).unsqueeze(0) * eps
        log_prob = gaussian_log_prob(raw_action, mean.unsqueeze(0), log_std_s.unsqueeze(0))
        velocity = raw_to_velocity(raw_action, v_min=v_min, v_max=v_max)
        return {
            "raw_action": raw_action.contiguous(),
            "velocity": velocity.contiguous(),
            "log_prob": log_prob.contiguous(),
            "mean": mean,
            "log_std": log_std,
        }

    def log_prob(self, x: torch.Tensor, raw_action: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        mean, log_std = self.forward(x)
        log_std_s = log_std + math.log(max(float(temperature), 1e-8))
        return gaussian_log_prob(raw_action, mean.unsqueeze(0), log_std_s.unsqueeze(0)).contiguous()

    @torch.no_grad()
    def initialize_mean_from_velocity(
        self,
        ctrl_velocity: torch.Tensor,
        shot_input: torch.Tensor,
        v_min: float,
        v_max: float,
    ) -> float:
        """Fit the final mean bias toward a known control grid for warm starts.

        This is intentionally a light-touch initializer for single-observation
        inversion. It does not train on the target during RL; it only aligns the
        output scale if a prior control grid is provided.
        """
        raw = velocity_to_raw(ctrl_velocity.to(shot_input.device), v_min, v_max)
        mean, _ = self.forward(shot_input)
        delta = raw.view_as(mean) - mean
        self.mean_head.bias.add_(delta.mean().view_as(self.mean_head.bias))
        return float(delta.abs().mean().detach().cpu().item())


class LearnableContinuousSplinePolicy(nn.Module):
    """Case-specific Gaussian policy with no seismic encoder.

    Useful as a pure neural parameterization of the search space for one
    observation, and as a baseline against the seismic-conditioned policy.
    """

    def __init__(
        self,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        init_velocity: torch.Tensor | None = None,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
        init_log_std: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.nx_ctrl = int(nx_ctrl)
        self.nz_ctrl = int(nz_ctrl)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        if init_velocity is None:
            init_raw = torch.zeros(self.nx_ctrl, self.nz_ctrl, dtype=torch.float32)
        else:
            init_raw = velocity_to_raw(init_velocity.float(), v_min, v_max)
        self.mean = nn.Parameter(init_raw.reshape(self.nx_ctrl, self.nz_ctrl).clone())
        self.log_std = nn.Parameter(torch.full((self.nx_ctrl, self.nz_ctrl), float(init_log_std)))

    def forward(self, x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.unsqueeze(0)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std).unsqueeze(0)
        if x is not None:
            b = int(x.shape[0])
            mean = mean.expand(b, -1, -1)
            log_std = log_std.expand(b, -1, -1)
        return mean.contiguous(), log_std.contiguous()

    def sample(
        self,
        x: torch.Tensor | None,
        n: int,
        *,
        v_min: float,
        v_max: float,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.forward(x)
        log_std_s = log_std + math.log(max(float(temperature), 1e-8))
        eps = torch.randn((int(n),) + tuple(mean.shape), device=mean.device, dtype=mean.dtype)
        raw_action = mean.unsqueeze(0) + torch.exp(log_std_s).unsqueeze(0) * eps
        log_prob = gaussian_log_prob(raw_action, mean.unsqueeze(0), log_std_s.unsqueeze(0))
        velocity = raw_to_velocity(raw_action, v_min=v_min, v_max=v_max)
        return {
            "raw_action": raw_action.contiguous(),
            "velocity": velocity.contiguous(),
            "log_prob": log_prob.contiguous(),
            "mean": mean,
            "log_std": log_std,
        }

    def log_prob(self, x: torch.Tensor | None, raw_action: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        mean, log_std = self.forward(x)
        log_std_s = log_std + math.log(max(float(temperature), 1e-8))
        return gaussian_log_prob(raw_action, mean.unsqueeze(0), log_std_s.unsqueeze(0)).contiguous()

