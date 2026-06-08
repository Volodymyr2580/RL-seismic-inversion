import math

import torch

from agents.fwi_rewards import (
    _ncc_maxlag_fft,
    reward_ncc_maxlag,
    reward_ncc_zero,
)
from agents.seismic_layout import as_shot_receiver_time
from agents.traveltime_reward import traveltime_reward


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


def test_forward_layout_normalizer_accepts_deepwave_variants():
    canonical = torch.randn(2, 3, 16)       # [shot, receiver, time]
    transposed = canonical.permute(0, 2, 1) # [shot, time, receiver]

    out1 = as_shot_receiver_time(canonical, n_receivers=3, nt=16)
    out2 = as_shot_receiver_time(transposed, n_receivers=3, nt=16)

    assert out1.shape == (2, 3, 16)
    assert out2.shape == (2, 3, 16)
    assert torch.allclose(out1, canonical)
    assert torch.allclose(out2, canonical)


def test_trace_rewards_reject_likely_receiver_time_swap():
    obs = _wavelet(160).reshape(1, 1, -1).repeat(1, 4, 1)
    pred_wrong = obs.unsqueeze(0).permute(0, 1, 3, 2)  # [G, shot, time, receiver]
    obs_wrong = obs.permute(0, 2, 1)                  # [shot, time, receiver]

    try:
        reward_ncc_zero(pred_wrong, obs_wrong)
    except ValueError as exc:
        assert "swapped" in str(exc)
    else:
        raise AssertionError("NCC reward accepted receiver/time-swapped data")


def test_traveltime_reward_is_nonzero_for_shifted_canonical_traces():
    nt = 240
    obs_trace = torch.zeros(nt)
    pred_trace = torch.zeros(nt)
    obs_trace[150] = 1.0
    pred_trace[180] = 1.0
    obs = obs_trace.reshape(1, 1, -1).repeat(1, 3, 1)
    pred = pred_trace.reshape(1, 1, 1, -1).repeat(1, 1, 3, 1)

    reward = traveltime_reward(pred, obs)

    assert reward.shape == (1,)
    assert reward.item() < -0.001


def test_traveltime_reward_rejects_receiver_time_swap():
    obs_trace = _wavelet(160).reshape(1, 1, -1).repeat(1, 4, 1)
    pred_wrong = obs_trace.unsqueeze(0).permute(0, 1, 3, 2)
    obs_wrong = obs_trace.permute(0, 2, 1)

    try:
        traveltime_reward(pred_wrong, obs_wrong)
    except ValueError as exc:
        assert "swapped" in str(exc)
    else:
        raise AssertionError("TT reward accepted receiver/time-swapped data")
