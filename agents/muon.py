"""
Muon optimizer: Momentum + Newton-Schulz orthogonalization.
Reference: Keller Jordan et al., "Muon: An optimizer for matrix parameters"
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute G / sqrt(G^T G + eps*I)."""
    assert G.ndim == 2, f"Muon expects 2D matrix, got {G.ndim}D"
    # Scale for numerical stability
    norm = G.norm()
    if norm < eps:
        return G
    X = G / (norm + eps)
    
    if X.shape[0] < X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    
    for _ in range(steps):
        A = X @ X.T
        X = 1.5 * X - 0.5 * X @ A
    
    if transposed:
        X = X.T
    return X * norm


class MuonState:
    """Per-parameter state for Muon."""
    def __init__(self):
        self.m: dict[int, torch.Tensor] = {}  # momentum buffer by param id
    
    def get_m(self, param: nn.Parameter) -> torch.Tensor:
        pid = id(param)
        if pid not in self.m:
            self.m[pid] = torch.zeros_like(param)
        return self.m[pid]


def muon_step(
    params: list[nn.Parameter],
    lr: float,
    state: MuonState,
    *,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.0,
) -> None:
    """Single Muon optimization step."""
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        
        # Weight decay
        if weight_decay > 0:
            g = g + weight_decay * p.data
        
        # Momentum
        m = state.get_m(p)
        m.mul_(momentum).add_(g, alpha=1 - momentum)
        
        # Nesterov
        update = g.add(m, alpha=momentum) if nesterov else m.detach().clone()
        
        # Newton-Schulz for 2D parameters
        if update.ndim == 2 and update.shape[0] > 1 and update.shape[1] > 1:
            update = newton_schulz(update, steps=ns_steps)
        
        # Apply
        p.data.add_(update, alpha=-lr)
