"""Utilities for enforcing seismic tensor layout contracts.

The training pipeline uses one canonical layout:

    single model: [shot, receiver, time]
    model batch:  [G, shot, receiver, time]

Deepwave versions/configurations can expose receiver amplitudes as either
[shot, receiver, time] or [shot, time, receiver], so forward wrappers should
normalize immediately before rewards see the data.
"""

from __future__ import annotations

import torch


def as_shot_receiver_time(
    data: torch.Tensor,
    *,
    n_receivers: int,
    nt: int,
    name: str = "seismic data",
) -> torch.Tensor:
    """Return data as [shot, receiver, time], transposing if needed."""
    if data.ndim != 3:
        raise ValueError(f"{name} must be 3-D [shot, receiver, time], got {tuple(data.shape)}")

    expected = (int(n_receivers), int(nt))
    trailing = tuple(int(x) for x in data.shape[-2:])
    if trailing == expected:
        return data.contiguous()

    transposed = (int(nt), int(n_receivers))
    if trailing == transposed:
        return data.permute(0, 2, 1).contiguous()

    raise ValueError(
        f"{name} has unsupported trailing shape {trailing}; expected "
        f"[receiver,time]={expected} or [time,receiver]={transposed}"
    )


def assert_shot_receiver_time(
    data: torch.Tensor,
    *,
    n_shots: int,
    n_receivers: int,
    nt: int,
    name: str = "seismic data",
) -> None:
    expected = (int(n_shots), int(n_receivers), int(nt))
    if tuple(int(x) for x in data.shape) != expected:
        raise ValueError(f"{name} must be [shot, receiver, time]={expected}, got {tuple(data.shape)}")


def assert_batch_shot_receiver_time(
    data: torch.Tensor,
    *,
    group_size: int,
    n_shots: int,
    n_receivers: int,
    nt: int,
    name: str = "seismic batch",
) -> None:
    expected = (int(group_size), int(n_shots), int(n_receivers), int(nt))
    if tuple(int(x) for x in data.shape) != expected:
        raise ValueError(f"{name} must be [G, shot, receiver, time]={expected}, got {tuple(data.shape)}")


def reject_likely_receiver_time_swap(data: torch.Tensor, *, name: str = "seismic data") -> None:
    """Fail fast when a tensor likely has receiver/time axes swapped.

    This is intentionally heuristic because reward functions do not know the
    survey geometry. In this project nt is much larger than n_receivers
    (1000 vs 70), so a last axis shorter than the preceding axis is a strong
    signal that data is [shot, time, receiver] instead of canonical.
    """
    if data.ndim >= 3 and int(data.shape[-1]) < int(data.shape[-2]):
        raise ValueError(
            f"{name} looks like receiver/time axes are swapped: got {tuple(data.shape)}. "
            "Expected the time axis to be last, e.g. [shot, receiver, time]."
        )
