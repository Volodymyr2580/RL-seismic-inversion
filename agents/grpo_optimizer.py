from __future__ import annotations

import torch
import torch.nn.functional as F


def log_prob_from_logits(logits: torch.Tensor, actions: torch.Tensor, *, temperature: float = 1.0, eps: float = 0.0):
    if logits.ndim != 4:
        raise ValueError(f"logits 期望 [B,nx,nz,K]，得到 {tuple(logits.shape)}")
    if actions.ndim != 3:
        raise ValueError(f"actions 期望 [B,nx,nz]，得到 {tuple(actions.shape)}")
    b, nx, nz, k = logits.shape
    if actions.shape != (b, nx, nz):
        raise ValueError(f"actions shape 与 logits 不一致：{tuple(actions.shape)} vs {(b, nx, nz)}")
    temp = float(temperature)
    if temp <= 0:
        raise ValueError(f"temperature 必须 > 0，得到 {temp}")
    eps_f = float(eps)
    if not (0.0 <= eps_f < 1.0):
        raise ValueError(f"eps 必须在 [0,1) 内，得到 {eps_f}")

    probs = F.softmax(logits / temp, dim=-1)
    if eps_f > 0.0:
        probs = probs * (1.0 - eps_f) + (eps_f / float(k))
    logp = torch.log(probs.clamp_min(1e-12))
    return logp.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(-1).contiguous()


def normalize_advantages(rewards: torch.Tensor, eps: float = 1e-8):
    if rewards.ndim != 2:
        raise ValueError(f"rewards 期望 [G,B]，得到 {tuple(rewards.shape)}")
    mean = rewards.mean(dim=0, keepdim=True)
    std = rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (rewards - mean) / std


def grpo_loss_from_logp(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
):
    if logp_new.shape != logp_old.shape:
        raise ValueError(f"logp_new/logp_old shape 不一致：{tuple(logp_new.shape)} vs {tuple(logp_old.shape)}")
    if logp_new.ndim != 4:
        raise ValueError(f"logp_* 期望 [G,B,nx,nz]，得到 {tuple(logp_new.shape)}")
    if advantages.ndim != 2:
        raise ValueError(f"advantages 期望 [G,B]，得到 {tuple(advantages.shape)}")
    g, b, nx, nz = logp_new.shape
    if advantages.shape != (g, b):
        raise ValueError(f"advantages shape 不匹配：{tuple(advantages.shape)} vs {(g, b)}")

    diff = (logp_new - logp_old).mean(dim=(2, 3))
    ratio = torch.exp(diff)
    clipped = torch.clamp(ratio, 1.0 - float(epsilon_low), 1.0 + float(epsilon_high))
    surr1 = ratio * advantages
    surr2 = clipped * advantages
    loss = -torch.mean(torch.minimum(surr1, surr2))
    return loss


def bins_to_velocity(actions: torch.Tensor, v_min: float, v_max: float, n_bins: int):
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3:
        raise ValueError(f"actions 期望 [B,nx,nz] 或 [nx,nz]，得到 {tuple(actions.shape)}")
    idx = actions.long().clamp(0, int(n_bins) - 1)
    bin_width = (float(v_max) - float(v_min)) / float(int(n_bins))
    return float(v_min) + (idx.float() + 0.5) * bin_width


def velocity_to_bins(velocity: torch.Tensor, v_min: float, v_max: float, n_bins: int):
    if not torch.is_tensor(velocity):
        velocity = torch.as_tensor(velocity)
    bin_width = (float(v_max) - float(v_min)) / float(int(n_bins))
    idx = torch.floor((velocity - float(v_min)) / bin_width).long()
    return idx.clamp(0, int(n_bins) - 1)

