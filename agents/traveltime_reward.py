"""
Travel-time reward — vectorised energy-ratio first-arrival picking.
"""

from __future__ import annotations
import torch


def first_arrival_energy_ratio(
    p: torch.Tensor, dt: float = 0.001,
    win: int = 20, long_ratio: int = 5, min_idx: int = 50,
    threshold_factor: float = 3.0,
) -> torch.Tensor:
    """
    Pick first arrival times for all traces.  p: [n_traces, nt] → [n_traces] (seconds).

    Uses Short-Term Average / Long-Term Average (STA/LTA) via cumulative sum for
    vectorisation — a single conv1d style pass instead of a per-sample inner loop.
    """
    assert p.ndim == 2, f"expected [n_traces, nt], got shape {p.shape}"
    n_traces, nt = p.shape
    device = p.device
    long_win = long_ratio * win
    min_idx = max(min_idx, long_win + win)

    # --- vectorised energy ratio via prefix sums ---
    energy = p.pow(2)                                              # [n_traces, nt]
    ecum = torch.cat([torch.zeros(n_traces, 1, device=device),     # prefix sum
                       torch.cumsum(energy, dim=1)], dim=1)         # [n_traces, nt+1]

    e_short = ecum[:, win:] - ecum[:, :-win]                       # [n_traces, nt-win+1]
    e_long = (ecum[:, long_win + win:] - ecum[:, :-long_win - win]
              ) + 1e-10                                             # [n_traces, nt-long_win-win+1]

    # Align: STA/LTA defined at sample t, short is [t-win, t), long is [t-long_win-win, t-win)
    align_start = long_win + win
    ratio = e_short[:, align_start - win:] / e_long                # [n_traces, nt-align_start]

    # Pad front with zeros so indices match original time axis
    ratio = torch.cat([torch.zeros(n_traces, align_start, device=device), ratio], dim=1)

    # --- threshold crossing ---
    thresh = threshold_factor * ratio.mean(dim=1, keepdim=True)    # [n_traces, 1]
    ratio[:, :min_idx] = 0.0                                       # mask early samples

    # Find first index where ratio > thresh for each trace
    mask = ratio > thresh
    first_idx = mask.int().argmax(dim=1)                           # [n_traces]
    # argmax returns 0 when no crossing — detect those
    no_crossing = ~mask.any(dim=1)
    if no_crossing.any():
        # Fallback: 20% of peak ratio
        max_ratio = ratio.max(dim=1).values                        # [n_traces]
        thresh2 = 0.2 * max_ratio
        mask2 = ratio > thresh2.unsqueeze(1)
        mask2[:, :min_idx] = False
        fallback_idx = mask2.int().argmax(dim=1)
        first_idx[no_crossing] = fallback_idx[no_crossing]
        # If still 0, use min_idx
        still_zero = (first_idx < min_idx)
        first_idx[still_zero] = min_idx

    return first_idx.float() * dt


def traveltime_reward(
    p_pred: torch.Tensor, p_obs: torch.Tensor, dt: float = 0.001,
) -> torch.Tensor:
    """
    Travel-time reward.  Returns [G] tensor (0 = perfect).

    Accepts two layouts (auto-detected):
      • [G, n_shots, nt, n_receivers]  — from simulate_batch  (4-D)
      • [G, nt, n_receivers]            — single-shot fallback (3-D)

    p_obs: [n_shots, nt, n_receivers] — single-shot ground truth.
    """
    if p_pred.ndim == 4:
        G, n_shots, nt, nr = p_pred.shape
        p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(-1, nt)    # [G*n_shots*nr, nt]
    elif p_pred.ndim == 3:
        G, nt, nr = p_pred.shape
        n_shots = 1
        p_pred_2d = p_pred.transpose(1, 2).reshape(-1, nt)         # [G*nr, nt]
    else:
        raise ValueError(f"p_pred must be 3-D or 4-D, got {p_pred.shape}")

    if p_obs.ndim == 3:
        _, nt_obs, nr_obs = p_obs.shape
        p_obs_2d = p_obs.permute(0, 2, 1).reshape(-1, nt_obs)     # [n_shots*nr, nt]
    elif p_obs.ndim == 2:
        p_obs_2d = p_obs  # already [n_traces, nt]
    else:
        raise ValueError(f"p_obs must be 2-D or 3-D, got {p_obs.shape}")

    t_obs = first_arrival_energy_ratio(p_obs_2d, dt=dt)            # [n_traces]
    n_traces = t_obs.shape[0]
    # n_traces already = n_shots * nr (all traces from all shots)
    traces_per_group = n_traces

    # Broadcast obs to each group: each group compares against same t_obs
    if p_pred.ndim == 4:
        t_obs = t_obs.unsqueeze(0).expand(G, -1).reshape(-1)      # [G * n_traces]

    rewards = torch.zeros(G, device=p_pred.device)
    for g in range(G):
        start = g * traces_per_group
        end = start + traces_per_group
        t_pred = first_arrival_energy_ratio(p_pred_2d[start:end], dt=dt)
        t_obs_g = t_obs[start:end]
        rewards[g] = -(t_pred - t_obs_g).abs().mean()

    return rewards


def traveltime_reward_log(
    p_pred: torch.Tensor, p_obs: torch.Tensor, dt: float = 0.001,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Log-scaled travel-time reward: amplifies small |Δt| differences.

    R = -log(mean(|Δt|) + ε)  — larger is better (less error → less negative).
    This stretches the low-error regime where raw |Δt| differences are tiny.
    """
    raw = traveltime_reward(p_pred, p_obs, dt=dt)   # [G], values ≤ 0
    return -torch.log(-raw + eps)                     # [G], large positive for good fits
