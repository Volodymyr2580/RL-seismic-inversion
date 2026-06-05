from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .forward_simulator import OpenFWIVelFamilyGeometry, DeepwaveScalarForward2D


def sign_preserving_log(x: torch.Tensor, k: float = 3.0, c: float = 0.0, eps: float = 1e-6):
    return torch.sign(x) * torch.log(k * torch.abs(x) + c + eps)


def reward_l1_l2_log(pred_seismic: torch.Tensor, obs_seismic: torch.Tensor, k: float = 3.0, c: float = 0.0, eps: float = 1e-6):
    pred_t = sign_preserving_log(pred_seismic, k=k, c=c, eps=eps)
    obs_t = sign_preserving_log(obs_seismic, k=k, c=c, eps=eps)
    l1 = torch.abs(pred_t - obs_t).sum(dim=(1, 2, 3))
    l2 = ((pred_t - obs_t) ** 2).sum(dim=(1, 2, 3))
    return -(l1 + l2)


@dataclass
class RewardConfig:
    mode: str = "fwi"
    lambda_mix: float = 0.5
    log_k: float = 3.0
    log_c: float = 0.0


class RewardCalculator:
    def __init__(
        self,
        cfg: RewardConfig,
        geom: OpenFWIVelFamilyGeometry,
        si_callback_freq: int = 1,
        si_pml_width: tuple[int, int, int, int] = (20, 20, 20, 20),
        si_pml_freq: float = 15.0,
        si_cal_pad: Optional[list[int]] = None,
    ):
        self.cfg = cfg
        self.geom = geom
        self.forward = DeepwaveScalarForward2D(geom)
        self.si_callback_freq = int(si_callback_freq)
        self.si_pml_width = list(int(x) for x in si_pml_width)
        self.si_pml_freq = float(si_pml_freq)
        self.si_cal_pad = si_cal_pad

    def fwi(self, pred_seismic: torch.Tensor, obs_seismic: torch.Tensor):
        return reward_l1_l2_log(pred_seismic, obs_seismic, k=self.cfg.log_k, c=self.cfg.log_c)

    def si(self, v_model_2d: torch.Tensor, obs_seismic: torch.Tensor, device: str):
        if v_model_2d.ndim != 2:
            raise ValueError(f"v_model_2d 必须是 2D，得到 {tuple(v_model_2d.shape)}")
        if obs_seismic.ndim != 3:
            raise ValueError(f"obs_seismic 必须是 3D [shot,time,receiver]，得到 {tuple(obs_seismic.shape)}")

        try:
            from example_to_dyy.funcs.RTM import rtm_imaging_batch_all_forw
        except Exception as e:
            raise RuntimeError(f"无法导入 RTM 模块用于 SI 计算：{e}") from e

        dx = float(self.geom.dx)
        dt = float(self.geom.dt)
        freq = float(self.geom.freq)
        n_shots = int(self.geom.n_shots)
        n_receivers = int(self.geom.n_receivers)
        nt = int(self.geom.nt)

        if obs_seismic.shape != (n_shots, nt, n_receivers):
            raise ValueError(f"obs_seismic shape 期望 {(n_shots, nt, n_receivers)}，得到 {tuple(obs_seismic.shape)}")

        source_locations, receiver_locations, source_amplitudes = self.forward._get_cached_io(device)
        obsv_data_masked = obs_seismic.permute(0, 2, 1).contiguous()
        image = torch.zeros_like(v_model_2d, dtype=torch.float32, device=device)
        h = rtm_imaging_batch_all_forw(
            image=image,
            v_apply=v_model_2d.to(device=device, dtype=torch.float32),
            grid_spacing=[dx, dx],
            dt=dt,
            source_amplitudes=source_amplitudes.to(device),
            source_locations=source_locations.to(device),
            obsv_data_masked=obsv_data_masked.to(device),
            receiver_locations=receiver_locations.to(device),
            batch_size=max(1, n_shots),
            max_vel=None,
            pml_width=self.si_pml_width,
            pml_freq=self.si_pml_freq if self.si_pml_freq is not None else freq,
            callback_freq=self.si_callback_freq,
            illum=False,
            illum_mul=None,
            forw_dt2=False,
            cal_pad=self.si_cal_pad,
            outSI=True,
        )
        return torch.tensor(float(h), device=device, dtype=torch.float32)

    def compute(self, pred_seismic: torch.Tensor, obs_seismic: torch.Tensor, v_model_2d: torch.Tensor | None, device: str):
        mode = str(self.cfg.mode).lower()
        if mode == "fwi":
            return self.fwi(pred_seismic, obs_seismic)
        if mode == "si":
            if v_model_2d is None:
                raise ValueError("SI reward 需要 v_model_2d")
            b = int(pred_seismic.shape[0])
            out = []
            for i in range(b):
                out.append(self.si(v_model_2d[i], obs_seismic[i], device=device))
            return torch.stack(out, dim=0)
        if mode == "mix":
            r_fwi = self.fwi(pred_seismic, obs_seismic)
            if v_model_2d is None:
                raise ValueError("mix reward 需要 v_model_2d")
            b = int(pred_seismic.shape[0])
            r_si = []
            for i in range(b):
                r_si.append(self.si(v_model_2d[i], obs_seismic[i], device=device))
            r_si_t = torch.stack(r_si, dim=0)
            return r_fwi + float(self.cfg.lambda_mix) * r_si_t
        raise ValueError(f"不支持的 reward_mode: {self.cfg.mode}")

