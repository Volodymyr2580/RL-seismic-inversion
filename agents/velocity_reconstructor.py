from __future__ import annotations

import torch

from .Bspline import bspline2d_prolong


class VelocityReconstructor:
    def __init__(self, nx_model: int, nz_model: int, wl: int = 0, fixed_upper_layer: torch.Tensor | None = None):
        self.nx_model = int(nx_model)
        self.nz_model = int(nz_model)
        self.wl = int(wl)
        self.fixed_upper_layer = fixed_upper_layer

    def reconstruct(self, v_ctrl: torch.Tensor):
        if v_ctrl.ndim == 2:
            v_ctrl = v_ctrl.unsqueeze(0)
        if v_ctrl.ndim != 3:
            raise ValueError(f"v_ctrl 期望 [B,nx_ctrl,nz_ctrl] 或 [nx_ctrl,nz_ctrl]，得到 {tuple(v_ctrl.shape)}")
        if self.wl <= 0:
            return bspline2d_prolong(v_ctrl, (self.nx_model, self.nz_model)).contiguous()

        if self.fixed_upper_layer is None:
            raise ValueError("wl > 0 时需要提供 fixed_upper_layer（形状 [nx_model, wl] 或 [B, nx_model, wl]）")

        lower = bspline2d_prolong(v_ctrl, (self.nx_model, self.nz_model - self.wl)).contiguous()

        upper = self.fixed_upper_layer
        if upper.ndim == 2:
            upper = upper.unsqueeze(0)
        if upper.ndim != 3:
            raise ValueError(f"fixed_upper_layer 期望 [nx_model, wl] 或 [B, nx_model, wl]，得到 {tuple(upper.shape)}")
        if upper.shape[1] != self.nx_model or upper.shape[2] != self.wl:
            raise ValueError(f"fixed_upper_layer shape 不匹配：期望 [B,{self.nx_model},{self.wl}]，得到 {tuple(upper.shape)}")

        if upper.shape[0] == 1 and lower.shape[0] > 1:
            upper = upper.expand(lower.shape[0], -1, -1)
        if upper.shape[0] != lower.shape[0]:
            raise ValueError(f"batch size 不一致：upper={int(upper.shape[0])}, lower={int(lower.shape[0])}")

        upper = upper.to(device=lower.device, dtype=lower.dtype)
        return torch.cat([upper, lower], dim=2).contiguous()

