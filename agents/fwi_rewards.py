"""
FWI data-misfit reward variants — alternatives to raw L2.
All operate on [G, n_shots, nt, n_receivers] tensors, return [G] rewards.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
# Utility: Frequency-band filtering
# ═══════════════════════════════════════════════════════════════

def lowpass_filter(x: torch.Tensor, cutoff_hz: float, dt: float = 0.001) -> torch.Tensor:
    """Low-pass filter via FFT. x: [..., nt] → [..., nt]."""
    nt = x.shape[-1]
    freqs = torch.fft.rfftfreq(nt, dt).to(device=x.device)
    X = torch.fft.rfft(x, dim=-1)
    mask = torch.clamp((1.2 * cutoff_hz - freqs) / (0.2 * cutoff_hz + 1e-10), 0.0, 1.0)
    return torch.fft.irfft(X * mask, n=nt, dim=-1)


# ═══════════════════════════════════════════════════════════════
# 1. Hilbert Envelope misfit
# ═══════════════════════════════════════════════════════════════

def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """Hilbert envelope via full complex FFT. x: [..., nt] → [..., nt].
    
    Uses full FFT + IFFT (not rfft/irfft) because the analytic signal
    is complex-valued and irfft cannot represent it.
    """
    n = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)                        # full complex spectrum [..., n]
    # Hilbert mask: zero negative freqs, double positive, keep DC/Nyquist
    h = torch.zeros(n, device=x.device, dtype=X.dtype)
    h[0] = 1.0
    if n % 2 == 0:
        h[1:n//2] = 2.0
        h[n//2] = 1.0                                   # Nyquist
    else:
        h[1:(n+1)//2] = 2.0
    analytic = torch.fft.ifft(X * h, dim=-1)            # complex analytic signal
    return analytic.abs()


def reward_envelope(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
) -> torch.Tensor:
    """Envelope L2 misfit: -||env(pred) - env(obs)||² per group."""
    G = p_pred.shape[0]
    # p_pred: [G, n_shots, nt, nr], p_obs: [n_shots, nt, nr]
    p_obs_batch = p_obs.unsqueeze(0).expand(G, -1, -1, -1)
    env_pred = hilbert_envelope(p_pred)
    env_obs = hilbert_envelope(p_obs_batch)
    return -((env_pred - env_obs) ** 2).sum(dim=(1, 2, 3))


# ═══════════════════════════════════════════════════════════════
# 2. Time-windowed L2 (TT picker + window around first arrival)
# ═══════════════════════════════════════════════════════════════

def reward_windowed_l2(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    *,
    dt: float = 0.001,
    win_before: int = 30,   # samples before first arrival
    win_after: int = 150,   # samples after first arrival
) -> torch.Tensor:
    """L2 misfit in a time window around the observed first arrival."""
    from agents.traveltime_reward import first_arrival_energy_ratio

    G, n_shots, nt, nr = p_pred.shape
    device = p_pred.device

    # Get TT picker reference (from obs, shared across groups)
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(-1, nt)  # [n_shots*nr, nt]
    t_obs = first_arrival_energy_ratio(p_obs_2d, dt=dt)  # [n_traces]

    n_traces = t_obs.shape[0]
    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, -1, nt)  # [G, n_traces, nt]

    rewards = torch.zeros(G, device=device)
    for g in range(G):
        total = 0.0
        for tr in range(n_traces):
            t0 = int(t_obs[tr] / dt)
            start = max(0, t0 - win_before)
            end = min(nt, t0 + win_after)
            diff = p_pred_2d[g, tr, start:end] - p_obs_2d[tr, start:end]
            total += (diff ** 2).sum().item()
        rewards[g] = -total
    return rewards


# ═══════════════════════════════════════════════════════════════
# 3. Wasserstein misfit (CDF-based, time-domain)
# ═══════════════════════════════════════════════════════════════
#
# Previous implementation sorted amplitudes and compared sorted values
# (= W₁ in AMPLITUDE space).  That destroyed all temporal/phase info:
# two identical signals shifted in time gave W₁=0, making the reward
# completely blind to arrival-time errors — fatal for FWI.
#
# Correct formulation (Métivier et al. 2016, Engquist & Froese 2014):
#  1. Convert signal to non-negative density on the TIME axis
#  2. Normalize to unit sum
#  3. W₁ = ∫ |CDF_pred(t) - CDF_obs(t)| dt
#
# This preserves sensitivity to time shifts while being convex w.r.t.
# translation, helping to avoid cycle-skipping.


def _to_density(x: torch.Tensor, normalize: str) -> torch.Tensor:
    """Convert oscillatory signal to non-negative density on the time axis.
    x: [..., nt] → [..., nt], all values ≥ 0.
    """
    if normalize == "abs":
        return x.abs()
    elif normalize == "square":
        return x ** 2
    elif normalize == "envelope":
        return hilbert_envelope(x)
    else:
        raise ValueError(f"Unknown normalize: {normalize}")


def _cdf_wasserstein1(
    p_pred: torch.Tensor, p_obs: torch.Tensor, *, normalize: str = "abs",
) -> torch.Tensor:
    """CDF-based W₁ in time domain. p_pred: [G, n_traces, nt], p_obs: [n_traces, nt] → [G, n_traces].
    
    Steps:
      signal → density → normalize → CDF → |CDF_diff|.sum() = W₁ (in sample units)
    """
    G, n_traces, nt = p_pred.shape
    device = p_pred.device

    # Step 1: non-negative density
    p_pred_nn = _to_density(p_pred, normalize)       # [G, n_traces, nt]
    p_obs_nn  = _to_density(p_obs, normalize)        # [n_traces, nt]

    # Step 2: normalize to probability distributions
    eps = 1e-10
    p_pred_prob = p_pred_nn / p_pred_nn.sum(dim=-1, keepdim=True).clamp(min=eps)
    p_obs_prob  = p_obs_nn  / p_obs_nn.sum(dim=-1, keepdim=True).clamp(min=eps)

    # Step 3: CDF
    cdf_pred = p_pred_prob.cumsum(dim=-1)            # [G, n_traces, nt]
    cdf_obs  = p_obs_prob.cumsum(dim=-1)             # [n_traces, nt]

    # Step 4: W₁ = sum(|CDF_diff|), in sample-index units
    # Broadcasting obs across groups
    w1 = (cdf_pred - cdf_obs.unsqueeze(0)).abs().sum(dim=-1)  # [G, n_traces]
    return w1


def reward_wasserstein(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    *,
    normalize: str = "abs",
) -> torch.Tensor:
    """Wasserstein-1 misfit in TIME domain (CDF-based), per trace, summed over shots & receivers.
    
    p_pred: [G, n_shots, nt, nr]   p_obs: [n_shots, nt, nr]
    normalize: "abs" | "square" | "envelope"
    Returns: [G]  (negative W₁, higher = better)
    """
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr

    # Reshape to per-trace 1D signals
    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces, nt)  # [G, n_traces, nt]
    p_obs_2d  = p_obs.permute(0, 2, 1).reshape(n_traces, nt)          # [n_traces, nt]

    w1_per_trace = _cdf_wasserstein1(p_pred_2d, p_obs_2d, normalize=normalize)  # [G, n_traces]
    w_groups = w1_per_trace.sum(dim=1)  # [G]
    return -w_groups


# ═══════════════════════════════════════════════════════════════
# 3b. Standard OT — W₂ (Métivier et al. 2016 formulation)
# ═══════════════════════════════════════════════════════════════
#
# The proper OT-FWI formulation from Métivier et al. (2016):
#   1. s̃(t) = (s(t)² + η) / ∫(s(t)² + η) dt   (η = small baseline shift)
#   2. Compute W₂ via inverse CDF (quantile function):
#      W₂² = ∫₀¹ |F⁻¹(α) − G⁻¹(α)|² dα
#
# For 1D this is the exact quadratic Wasserstein distance,
# unlike our CDF-based W₁ (Section 3) which uses L1 of CDF differences.


def _wasserstein2_1d(
    p_pred: torch.Tensor, p_obs: torch.Tensor, *,
    eta: float = 1e-3, n_quantiles: int = 500,
) -> torch.Tensor:
    """Standard W₂ via quantile functions. p_pred: [G, T, nt], p_obs: [T, nt] → [G, T].
    
    Métivier et al. 2016: s̃ = (s² + η) / ∫(s² + η) → normalize → W₂ via F⁻¹.
    """
    G, n_traces, nt = p_pred.shape
    device = p_pred.device

    # Step 1: non-negative density (squared + baseline)
    p_pred_nn = p_pred ** 2 + eta                       # [G, T, nt]
    p_obs_nn  = p_obs ** 2 + eta                        # [T, nt]

    # Step 2: normalize to probability distributions
    eps = 1e-10
    p_pred_prob = p_pred_nn / p_pred_nn.sum(dim=-1, keepdim=True).clamp(min=eps)
    p_obs_prob  = p_obs_nn  / p_obs_nn.sum(dim=-1, keepdim=True).clamp(min=eps)

    # Step 3: CDF
    cdf_pred = p_pred_prob.cumsum(dim=-1)               # [G, T, nt]
    cdf_obs  = p_obs_prob.cumsum(dim=-1)                # [T, nt]

    # Step 4: W₂ via quantile (inverse CDF)
    alpha = torch.linspace(0.0, 1.0, n_quantiles + 1, device=device)[1:]  # (0,1]
    # searchsorted returns the first index where cdf >= alpha
    # cdf is [G, T, nt] or [T, nt], alpha is [n_quantiles]
    q_pred = torch.searchsorted(
        cdf_pred.reshape(G * n_traces, nt).contiguous(),
        alpha.unsqueeze(0).expand(G * n_traces, -1).contiguous()
    ).float().reshape(G, n_traces, n_quantiles)         # [G, T, Q]

    q_obs = torch.searchsorted(
        cdf_obs.reshape(n_traces, nt).contiguous(),
        alpha.unsqueeze(0).expand(n_traces, -1).contiguous()
    ).float().reshape(1, n_traces, n_quantiles)          # [1, T, Q]

    # W₂² = mean((q_pred - q_obs)²)  (in sample-index units, squared)
    w2_sq = ((q_pred - q_obs) ** 2).mean(dim=-1)        # [G, T]
    w2 = w2_sq.sqrt()                                    # [G, T]  W₂ in sample units
    return w2


def reward_wasserstein_w2(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    *,
    eta: float = 1e-3,
) -> torch.Tensor:
    """Standard OT W₂ misfit (Métivier et al. 2016), per trace, summed.
    
    p_pred: [G, n_shots, nt, nr]   p_obs: [n_shots, nt, nr]
    Returns: [G]  (negative W₂, higher = better)
    """
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr

    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces, nt)
    p_obs_2d  = p_obs.permute(0, 2, 1).reshape(n_traces, nt)

    w2_per_trace = _wasserstein2_1d(p_pred_2d, p_obs_2d, eta=eta)  # [G, n_traces]
    w_groups = w2_per_trace.sum(dim=1)  # [G]
    return -w_groups


# ═══════════════════════════════════════════════════════════════
# 4. Contrastive reward (spectrum similarity + trace CC)
# ═══════════════════════════════════════════════════════════════

def spectrum_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of magnitude spectra. x, y: [B, N] → [B]."""
    X = torch.fft.rfft(x.contiguous(), dim=-1).abs()
    Y = torch.fft.rfft(y.contiguous(), dim=-1).abs()
    # Cosine similarity
    dot = (X * Y).sum(dim=-1)
    norm_x = X.norm(dim=-1)
    norm_y = Y.norm(dim=-1)
    return dot / (norm_x * norm_y + 1e-10)


def trace_corr_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Max normalized cross-correlation. x, y: [B, N] → [B]."""
    # Normalize
    x_n = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-10)
    y_n = (y - y.mean(dim=-1, keepdim=True)) / (y.std(dim=-1, keepdim=True) + 1e-10)
    # Cross-correlation via FFT
    N = x.shape[-1]
    X = torch.fft.rfft(x_n.contiguous(), n=2*N, dim=-1)
    Y = torch.fft.rfft(y_n.contiguous(), n=2*N, dim=-1)
    corr = torch.fft.irfft(X * Y.conj(), n=2*N, dim=-1)[..., :N] / N
    return corr.max(dim=-1).values


def reward_contrastive(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    *,
    spec_weight: float = 0.5,
    cc_weight: float = 0.5,
) -> torch.Tensor:
    """Contrastive reward: spectrum similarity + max cross-correlation."""
    G, n_shots, nt, nr = p_pred.shape
    n_traces_per_group = n_shots * nr
    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces_per_group, nt).contiguous()
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(-1, nt).contiguous()

    rewards = torch.zeros(G, device=p_pred.device)
    for g in range(G):
        spec = spectrum_similarity(p_pred_2d[g], p_obs_2d).mean()
        cc = trace_corr_max(p_pred_2d[g], p_obs_2d).mean()
        rewards[g] = spec_weight * spec + cc_weight * cc
    return rewards


# ═══════════════════════════════════════════════════════════════
# 5. NCC rewards: normalized cross-correlation variants
# ═══════════════════════════════════════════════════════════════
#
# These shift the misfit from "are amplitudes equal?" (L2) to
# "are waveforms similarly shaped?" (NCC), reducing sensitivity
# to amplitude scaling / source-strength errors.
#
# R1: NCC_zero   — zero-lag only, simplest robust baseline
# R2: NCC_maxlag — max over lag window + explicit lag penalty


def _ncc_zero(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Normalized zero-lag cross-correlation. x, y: [B, N] → [B] in [-1, 1]."""
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    dot = (x * y).sum(dim=-1)
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    return dot / (nx * ny + 1e-10)


def _ncc_maxlag_fft(
    x: torch.Tensor, y: torch.Tensor, lag_max: int = 80,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max NCC and optimal lag via FFT. x, y: [B, N] → (ncc_max[B], lag[B]).
    
    ncc_max: best normalized CC value in [-lag_max, lag_max]
    lag:     optimal shift in samples (negative = x leads y)
    """
    B, N = x.shape
    # Zero-mean, unit-norm
    xm = x - x.mean(dim=-1, keepdim=True)
    ym = y - y.mean(dim=-1, keepdim=True)
    xm = xm / (xm.norm(dim=-1, keepdim=True) + 1e-10)
    ym = ym / (ym.norm(dim=-1, keepdim=True) + 1e-10)

    # Cross-correlation via FFT: r[τ] = IFFT(conj(FFT(y)) * FFT(x))
    n_fft = 2 * N
    X = torch.fft.rfft(xm, n=n_fft, dim=-1)
    Y = torch.fft.rfft(ym, n=n_fft, dim=-1)
    corr_full = torch.fft.irfft(X * Y.conj(), n=n_fft, dim=-1)  # [B, n_fft]
    # corr_full[τ] for τ ∈ [0, N-1] then negative wrapped

    # Shift zero-lag to center for easier indexing, or build index range
    # Positive lag (y delayed): corr_full[lag]  for lag in [0, lag_max]
    # Negative lag (y leads):  corr_full[n_fft + lag] for lag in [-lag_max, -1]
    n_fft_t = corr_full.shape[-1]

    # Build combined correlation values for lags in [-lag_max, lag_max]
    vals = []
    for lag in range(-lag_max, lag_max + 1):
        if lag >= 0:
            idx = lag
        else:
            idx = n_fft_t + lag
        vals.append(corr_full[:, idx:idx+1])
    corr_window = torch.cat(vals, dim=-1)  # [B, 2*lag_max+1]

    ncc_max, pos = corr_window.max(dim=-1)   # [B]
    lag_optimal = pos - lag_max                # [B], in [-lag_max, lag_max]
    return ncc_max, lag_optimal


def reward_ncc_zero(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
) -> torch.Tensor:
    """R1: Normalized zero-lag cross-correlation, mean over traces.
    
    Range: [-1, 1] per trace, averaged → higher = better waveform shape match.
    """
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr

    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces, nt)  # [G, T, nt]
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(1, n_traces, nt)       # [1, T, nt]

    # Compute per-trace NCC
    ncc = _ncc_zero(p_pred_2d.reshape(G * n_traces, nt),
                    p_obs_2d.expand(G, -1, -1).reshape(G * n_traces, nt))
    ncc = ncc.reshape(G, n_traces)  # [G, n_traces]

    return ncc.mean(dim=1)  # [G] in [-1, 1]


def reward_ncc_maxlag(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    *,
    lag_max: int = 80,
    lag_penalty: float = 0.005,
) -> torch.Tensor:
    """R2: Maximum NCC over lag window + lag penalty.
    
    For each trace, finds best alignment within ±lag_max samples.
    Reward = mean(ncc_max - penalty * |lag_optimal|).
    
    Encourages both high correlation AND zero time-lag.
    """
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr
    device = p_pred.device

    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces, nt)  # [G, T, nt]
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(1, n_traces, nt)       # [1, T, nt]

    all_ncc = []
    all_penalty = []
    for g in range(G):
        ncc_max, lag_opt = _ncc_maxlag_fft(
            p_pred_2d[g], p_obs_2d[0], lag_max=lag_max
        )  # ncc_max: [n_traces], lag_opt: [n_traces]
        all_ncc.append(ncc_max)
        all_penalty.append(lag_penalty * lag_opt.abs().float())

    ncc = torch.stack(all_ncc, dim=0)           # [G, n_traces]
    penalty = torch.stack(all_penalty, dim=0)    # [G, n_traces]

    return (ncc - penalty).mean(dim=1)  # [G]


# ═══════════════════════════════════════════════════════════════
# 6. Envelope NCC — Oh & Alkhalifah 2018
# ═══════════════════════════════════════════════════════════════
#
# Replaces Envelope L2 (which still compares envelope amplitudes)
# with envelope NCC — only compares envelope shape.
# Envelope produces artificial low frequencies, helping recover
# long-wavelength background velocity.


def reward_envelope_ncc(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
) -> torch.Tensor:
    """R3: Envelope NCC — Hilbert envelope + zero-lag NCC.
    
    Steps: signal → envelope → zero-mean → NCC → mean over traces.
    Range: [-1, 1], higher = better envelope shape match.
    """
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr

    # Compute envelopes
    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces, nt)
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(1, n_traces, nt)

    env_pred = hilbert_envelope(p_pred_2d)    # [G, T, nt]
    env_obs = hilbert_envelope(p_obs_2d)       # [1, T, nt]

    # NCC on envelopes
    ncc = _ncc_zero(env_pred.reshape(G * n_traces, nt),
                    env_obs.expand(G, -1, -1).reshape(G * n_traces, nt))
    ncc = ncc.reshape(G, n_traces)

    return ncc.mean(dim=1)  # [G]


# ═══════════════════════════════════════════════════════════════
# 7. AWI — Adaptive Waveform Inversion (Warner & Guasch 2016)
# ═══════════════════════════════════════════════════════════════
#
# Instead of comparing d_pred and d_obs directly, compute the
# Wiener matching filter w such that w * d_pred ≈ d_obs.
# If the velocity model is correct, w should be a delta at zero-lag.
# Reward penalizes the time-spread of w.


def _awi_matching_filter(
    d_pred: torch.Tensor, d_obs: torch.Tensor, eps: float = 1e-3,
) -> torch.Tensor:
    """Wiener matching filter in frequency domain.
    
    Find w such that w * d_pred ≈ d_obs.
    d_pred, d_obs: [B, N] → w: [B, N]  (time-domain, fftshift'd so zero-lag is center)
    """
    B, N = d_pred.shape
    # FFT
    D_pred = torch.fft.rfft(d_pred, dim=-1)   # [B, N//2+1]
    D_obs = torch.fft.rfft(d_obs, dim=-1)

    # Wiener filter: W = D_obs * conj(D_pred) / (|D_pred|^2 + eps * max|D_pred|^2)
    # Denominator uses PREDICTED power (we're filtering pred to match obs)
    pow_pred = D_pred.real ** 2 + D_pred.imag ** 2
    reg = eps * pow_pred.max(dim=-1, keepdim=True).values
    W = (D_obs * D_pred.conj()) / (pow_pred + reg + 1e-12)

    # Inverse FFT → time-domain filter
    w_full = torch.fft.irfft(W, n=N, dim=-1)  # [B, N]
    # fftshift: zero-lag (τ=0) should be at the center, not at index 0
    w = torch.fft.fftshift(w_full, dim=-1)     # [B, N], center = zero-lag
    return w


def reward_awi(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    *,
    version: str = "l1",    # "l1" | "full"
    eps: float = 1e-3,
) -> torch.Tensor:
    """R4: AWI reward — penalize matching-filter time-spread.
    
    version='l1':   R = -mean(|t| * w[t]^2) / mean(w[t]^2)
                    (simpler, mixes shift + spread)
    version='full': R = -(α·|τ_center| + β·τ_spread)
                    (separates shift from spread, more principled)
    """
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr
    device = p_pred.device

    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G, n_traces, nt)
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(1, n_traces, nt)

    # Time index tensor for moment calculations (zero-centered)
    t_idx = torch.arange(nt, device=device, dtype=torch.float32)
    t_center = nt // 2
    t_shifted = t_idx - t_center  # [-N/2, ..., N/2-1]

    rewards = torch.zeros(G, device=device)
    for g in range(G):
        w = _awi_matching_filter(p_pred_2d[g], p_obs_2d[0], eps=eps)  # [T, nt]
        w2 = w ** 2                                                    # [T, nt]
        w2_sum = w2.sum(dim=-1).clamp(min=1e-10)                       # [T]

        if version == "l1":
            # Weighted L1 moment: mean(|t| * w^2) / mean(w^2)
            spread = (t_shifted.abs() * w2).sum(dim=-1) / w2_sum       # [T]
            r_per_trace = -spread
        elif version == "full":
            # Separate center and spread
            tau_center = (t_shifted * w2).sum(dim=-1) / w2_sum         # [T]
            tau_spread2 = ((t_shifted - tau_center.unsqueeze(-1)) ** 2 * w2).sum(dim=-1) / w2_sum
            tau_spread = tau_spread2.sqrt()
            # Combine: penalize both shift and spread
            r_per_trace = -(0.5 * tau_center.abs() + 0.5 * tau_spread)
        else:
            raise ValueError(f"Unknown AWI version: {version}")

        rewards[g] = r_per_trace.mean()

    return rewards


# ═══════════════════════════════════════════════════════════════
# 8. Phase-only reward — FFT → amp-norm → IFFT → L2
# ═══════════════════════════════════════════════════════════════


def _phase_only_signal(x: torch.Tensor) -> torch.Tensor:
    """Amplitude-normalized IFFT: FFT → |X|=1 → IFFT. x: [B, N] → [B, N]."""
    X = torch.fft.rfft(x, dim=-1)
    mag = X.abs().clamp(min=1e-10)
    X_norm = X / mag
    return torch.fft.irfft(X_norm, n=x.shape[-1], dim=-1)


def reward_phase_func(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
) -> torch.Tensor:
    """Phase-only L2 misfit: -||phase_only(pred) - phase_only(obs)||² per group."""
    G, n_shots, nt, nr = p_pred.shape
    n_traces = n_shots * nr
    p_pred_2d = p_pred.permute(0, 1, 3, 2).reshape(G * n_traces, nt)
    p_obs_2d = p_obs.permute(0, 2, 1).reshape(n_traces, nt)
    phase_pred = _phase_only_signal(p_pred_2d)
    phase_obs = _phase_only_signal(p_obs_2d)
    phase_obs_exp = phase_obs.unsqueeze(0).expand(G, -1, -1).reshape(G * n_traces, nt)
    diff2 = (phase_pred - phase_obs_exp) ** 2
    l2_per_group = diff2.sum(dim=-1).reshape(G, n_traces).sum(dim=1)
    return -l2_per_group


def compute_fwi_reward(
    p_pred: torch.Tensor, p_obs: torch.Tensor,
    fwi_type: str = "l2",
    **kwargs,
) -> torch.Tensor:
    """Compute FWI data-misfit reward by type. Returns [G]."""
    if fwi_type == "l2":
        from agents.rl_objectives import reward_l2
        p_obs_batch = p_obs.unsqueeze(0).expand(p_pred.shape[0], -1, -1, -1)
        return reward_l2(p_pred, p_obs_batch)
    elif fwi_type == "envelope":
        return reward_envelope(p_pred, p_obs)
    elif fwi_type == "windowed_l2":
        return reward_windowed_l2(p_pred, p_obs, **kwargs)
    elif fwi_type == "wasserstein":
        return reward_wasserstein(p_pred, p_obs, **kwargs)
    elif fwi_type == "contrastive":
        return reward_contrastive(p_pred, p_obs, **kwargs)
    elif fwi_type == "ncc_zero":
        return reward_ncc_zero(p_pred, p_obs)
    elif fwi_type == "ncc_maxlag":
        return reward_ncc_maxlag(p_pred, p_obs, **kwargs)
    elif fwi_type == "envelope_ncc":
        return reward_envelope_ncc(p_pred, p_obs)
    elif fwi_type == "awi":
        return reward_awi(p_pred, p_obs, **kwargs)
    elif fwi_type == "phase_func":
        return reward_phase_func(p_pred, p_obs, **kwargs)
    elif fwi_type == "wasserstein_w2":
        return reward_wasserstein_w2(p_pred, p_obs, **kwargs)
    else:
        raise ValueError(f"Unknown fwi_type: {fwi_type}")
