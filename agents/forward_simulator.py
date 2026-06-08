from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import deepwave

from agents.seismic_layout import as_shot_receiver_time


@dataclass(frozen=True)
class OpenFWIVelFamilyGeometry:
    dx: float = 10.0
    dt: float = 0.001
    freq: float = 15.0
    nbc: int = 120
    nt: int = 1000
    n_shots: int = 5
    n_receivers: int = 70

    def default_source_x(self):
        n_shots = int(self.n_shots)
        n_receivers = int(self.n_receivers)
        if n_shots <= 1:
            return [max(0, (n_receivers - 1) // 2)]
        xs = torch.linspace(0, n_receivers - 1, steps=n_shots)
        xs = torch.round(xs).to(dtype=torch.long).tolist()
        out = []
        last = None
        for x in xs:
            x = int(x)
            if last is not None and x == last:
                x = min(x + 1, n_receivers - 1)
            out.append(x)
            last = x
        return out

    def default_surface_z(self):
        return 0


class DeepwaveScalarForward2D:
    def __init__(self, geom: OpenFWIVelFamilyGeometry):
        self.geom = geom
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_cached_io(self, device: str):
        key = str(device)
        if key in self._cache:
            return self._cache[key]

        n_shots = int(self.geom.n_shots)
        n_src = 1
        n_rec = int(self.geom.n_receivers)
        nt = int(self.geom.nt)
        dt = float(self.geom.dt)
        freq = float(self.geom.freq)
        nbc = int(self.geom.nbc)

        src_x = self.geom.default_source_x()
        if len(src_x) != n_shots:
            raise ValueError("source_x 数量与 n_shots 不一致")

        surf_z = int(self.geom.default_surface_z())

        source_locations = torch.zeros(n_shots, n_src, 2, dtype=torch.long, device=device)
        source_locations[:, 0, 0] = torch.tensor([int(x) for x in src_x], dtype=torch.long, device=device)
        source_locations[:, 0, 1] = int(surf_z)

        receiver_locations = torch.zeros(n_shots, n_rec, 2, dtype=torch.long, device=device)
        receiver_locations[:, :, 0] = (torch.arange(n_rec, dtype=torch.long, device=device)).repeat(n_shots, 1)
        receiver_locations[:, :, 1] = int(surf_z)

        peak_time = 1.5 / freq
        source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, n_src, 1).to(device)

        self._cache[key] = (source_locations, receiver_locations, source_amplitudes)
        return self._cache[key]

    def simulate(self, v_model_2d: np.ndarray | torch.Tensor, device: str = "cpu"):
        if isinstance(v_model_2d, torch.Tensor):
            if v_model_2d.ndim != 2:
                raise ValueError(f"v_model_2d 必须是 2D，得到 shape={tuple(v_model_2d.shape)}")
            v = v_model_2d.to(device=device, dtype=torch.float32)
            nbc = int(self.geom.nbc)
            v4 = v.unsqueeze(0).unsqueeze(0)
            v4 = F.pad(v4, (nbc, nbc, nbc, nbc), mode="replicate")
            v = v4.squeeze(0).squeeze(0).contiguous()
        else:
            v_np = np.asarray(v_model_2d, dtype=np.float32)
            if v_np.ndim != 2:
                raise ValueError(f"v_model_2d 必须是 2D，得到 shape={v_np.shape}")
            nbc = int(self.geom.nbc)
            v_pad = np.pad(v_np, ((nbc, nbc), (nbc, nbc)), mode="edge")
            v = torch.from_numpy(v_pad).to(device=device, dtype=torch.float32)

        dx = float(self.geom.dx)
        dt = float(self.geom.dt)
        freq = float(self.geom.freq)
        source_locations, receiver_locations, source_amplitudes = self._get_cached_io(device)
        nbc = int(self.geom.nbc)
        src_loc = source_locations.clone()
        rec_loc = receiver_locations.clone()
        src_loc[..., 0] += nbc
        src_loc[..., 1] += nbc
        rec_loc[..., 0] += nbc
        rec_loc[..., 1] += nbc

        with torch.no_grad():
            out = deepwave.scalar(
                v,
                dx,
                dt,
                source_amplitudes=source_amplitudes,
                source_locations=src_loc,
                receiver_locations=rec_loc,
                pml_width=nbc,
                pml_freq=freq,
            )
            rec = as_shot_receiver_time(
                out[-1],
                n_receivers=self.geom.n_receivers,
                nt=self.geom.nt,
                name="deepwave receiver data",
            )
        return rec

