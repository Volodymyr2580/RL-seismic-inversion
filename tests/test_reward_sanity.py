import math

import torch

from agents.fwi_rewards import (
    _ncc_maxlag_fft,
    reward_ncc_maxlag,
    reward_ncc_zero,
)


def _wavelet(nt: int = 128) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, nt)
    a = (math.pi * 6.0 * t) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def _shift_right_zero(x: torch.Tensor, shift: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    out[..., shift:] = x[..., :-shift]
    return out


def test_ncc_zero_is_amplitude_invariant_on_canonical_layout():
    obs_trace = _wavelet()
    obs = obs_trace.reshape(1, 1, -1).repeat(2, 3, 1)  # [shot, receiver, time]
    pred = (3.5 * obs).unsqueeze(0)                    # [G, shot, receiver, time]

    reward = reward_ncc_zero(pred, obs)

    assert reward.shape == (1,)
    assert torch.allclose(reward, torch.ones_like(reward), atol=1e-5)


def test_ncc_zero_penalizes_time_shift_on_canonical_layout():
    obs_trace = _wavelet()
    obs = obs_trace.reshape(1, 1, -1).repeat(1, 2, 1)
    pred_same = obs.unsqueeze(0)
    pred_shifted = _shift_right_zero(obs, 12).unsqueeze(0)

    same_reward = reward_ncc_zero(pred_same, obs)
    shifted_reward = reward_ncc_zero(pred_shifted, obs)

    assert same_reward.item() > 0.99
    assert shifted_reward.item() < same_reward.item() - 0.1


def test_ncc_maxlag_finds_known_time_shift():
    obs = _wavelet().unsqueeze(0)
    shift = 17
    pred = _shift_right_zero(obs, shift)

    ncc, lag = _ncc_maxlag_fft(pred, obs, lag_max=32)

    assert ncc.item() > 0.9
    assert abs(int(lag.item())) == shift


def test_reward_ncc_maxlag_uses_last_axis_as_time():
    obs_trace = _wavelet(96)
    obs = obs_trace.reshape(1, 1, -1).repeat(1, 4, 1)  # [shot, receiver, time]
    pred = _shift_right_zero(obs, 9).unsqueeze(0)

    reward = reward_ncc_maxlag(pred, obs, lag_max=16, lag_penalty=0.0)

    assert reward.shape == (1,)
    assert reward.item() > 0.9
