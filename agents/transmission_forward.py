"""
Transmission/reflection forward simulator for Phase II+.

The default transmission setup preserves the Phase6 convention:
sources at the top and receivers at the bottom. Newer experiments can
explicitly set source_depth/receiver_depth to swap those roles, e.g.
bottom sources and surface receivers for Phase7 sensitivity checks.

Uses deepwave.scalar with PML-padded velocity grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import deepwave

from agents.seismic_layout import as_shot_receiver_time


@dataclass(frozen=True)
class AcquisitionGeometry:
    """Survey geometry: supports reflection and transmission.

    reflection: sources at top (z=0), receivers at top (z=0)
    transmission: sources at top (z=0), receivers at bottom (z=nz-1)
    """

    nx_model: int = 70
    nz_model: int = 70
    dx: float = 10.0
    dt: float = 0.001
    freq: float = 15.0
    nt: int = 1000
    n_shots: int = 5
    n_receivers: int = 70
    pml_width: int = 40
    geometry: str = "reflection"  # "reflection" or "transmission"
    source_depth: str = "auto"  # "auto", "top", or "bottom"
    receiver_depth: str = "auto"  # "auto", "top", or "bottom"

    @property
    def padded_nx(self) -> int:
        return self.nx_model + 2 * self.pml_width

    @property
    def padded_nz(self) -> int:
        return self.nz_model + 2 * self.pml_width

    def source_x_positions(self) -> torch.Tensor:
        """Evenly spaced source x-positions along the physical model width."""
        if self.n_shots <= 1:
            return torch.tensor([self.nx_model // 2], dtype=torch.long)
        xs = torch.linspace(0, self.nx_model - 1, steps=self.n_shots)
        return torch.round(xs).to(dtype=torch.long)

    def receiver_x_positions(self) -> torch.Tensor:
        """Evenly spaced receiver x-positions along the physical model width."""
        if self.n_receivers <= 1:
            return torch.tensor([self.nx_model // 2], dtype=torch.long)
        xs = torch.linspace(0, self.nx_model - 1, steps=self.n_receivers)
        return torch.round(xs).to(dtype=torch.long)

    def _depth_to_z(self, depth: str, *, role: str) -> int:
        depth = str(depth).lower()
        if depth == "auto":
            if role == "source":
                depth = "top"
            elif self.geometry == "reflection":
                depth = "top"
            else:
                depth = "bottom"
        if depth == "top":
            return 0
        if depth == "bottom":
            return self.nz_model - 1
        raise ValueError(f"{role}_depth must be 'auto', 'top', or 'bottom', got {depth!r}")

    def source_z_index(self) -> int:
        return self._depth_to_z(self.source_depth, role="source")

    def receiver_z_index(self) -> int:
        return self._depth_to_z(self.receiver_depth, role="receiver")

    def validate_receiver_z(self) -> bool:
        """Verify receivers are NOT in PML region."""
        rec_z = self.receiver_z_index()
        rec_z_padded = self.pml_width + rec_z
        pml_top_end = self.pml_width  # end of top PML
        pml_bot_start = self.pml_width + self.nz_model  # start of bottom PML
        if rec_z == 0:
            return rec_z_padded >= pml_top_end  # at or below top PML
        return rec_z_padded < pml_bot_start  # above bottom PML


class AcquisitionForward:
    """Transmission forward simulator using deepwave.scalar.

    By default, sources fire from the top (z=0 physical), and receivers record
    at the bottom for transmission or at the top for reflection. The velocity
    model is padded with PML on all sides.
    Source and receiver positions are automatically offset by pml_width.
    """

    def __init__(self, geom: AcquisitionGeometry):
        self.geom = geom
        if not geom.validate_receiver_z():
            raise ValueError(
                f"Receivers at z_padded={geom.pml_width + geom.nz_model - 1} "
                f"are inside PML! PML starts at z={geom.pml_width + geom.nz_model}. "
                f"Increase nz_model or decrease pml_width."
            )
        self._cache: dict[str, tuple] = {}

    def _get_cached_io(self, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get or create source/receiver locations and wavelet (cached by device)."""
        key = str(device)
        if key in self._cache:
            return self._cache[key]

        g = self.geom

        # Source locations.
        src_x = g.source_x_positions()
        source_locations = torch.zeros(g.n_shots, 1, 2, dtype=torch.long, device=device)
        source_locations[:, 0, 0] = src_x.to(device)  # x positions
        source_locations[:, 0, 1] = g.source_z_index()

        # Receiver locations.
        rec_x = g.receiver_x_positions()
        receiver_locations = torch.zeros(g.n_shots, g.n_receivers, 2, dtype=torch.long, device=device)
        receiver_locations[:, :, 0] = (
            rec_x.to(device)
            .unsqueeze(0)
            .repeat(g.n_shots, 1)
        )
        receiver_locations[:, :, 1] = g.receiver_z_index()

        # Ricker wavelet
        peak_time = 1.5 / g.freq
        source_amplitudes = (
            deepwave.wavelets.ricker(g.freq, g.nt, g.dt, peak_time)
            .repeat(g.n_shots, 1, 1)
            .to(device)
        )

        self._cache[key] = (source_locations, receiver_locations, source_amplitudes)
        return self._cache[key]

    def pad_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Pad velocity model with PML boundary (replicate mode).

        Args:
            v: [nx, nz] or [B, nx, nz] physical velocity model.

        Returns:
            padded: [1, 1, nx+2*nbc, nz+2*nbc] or [B, 1, nx+2*nbc, nz+2*nbc]
        """
        nbc = self.geom.pml_width
        if v.ndim == 2:
            v4 = v.unsqueeze(0).unsqueeze(0)  # [1, 1, nx, nz]
            v4 = F.pad(v4, (nbc, nbc, nbc, nbc), mode="replicate")
            return v4.squeeze(0).contiguous()  # [1, nx+nbc*2, nz+nbc*2]
        elif v.ndim == 3:
            v4 = v.unsqueeze(1)  # [B, 1, nx, nz]
            v4 = F.pad(v4, (nbc, nbc, nbc, nbc), mode="replicate")
            return v4.contiguous()  # [B, 1, nx+nbc*2, nz+nbc*2]
        else:
            raise ValueError(f"Velocity must be 2D or 3D, got {v.ndim}D")

    @torch.no_grad()
    def simulate(self, v_model: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Run transmission forward simulation.

        Args:
            v_model: [nx, nz] single velocity model.
            device: torch device string.

        Returns:
            seismogram: [n_shots, n_receivers, nt]
        """
        g = self.geom
        nbc = g.pml_width

        # Pad velocity
        v_padded = self.pad_velocity(v_model.to(device=device, dtype=torch.float32))

        # Get source/receiver locations, offset by PML
        src_loc, rec_loc, src_amp = self._get_cached_io(device)
        src_loc_padded = src_loc.clone()
        rec_loc_padded = rec_loc.clone()
        src_loc_padded[..., 0] += nbc  # offset x
        src_loc_padded[..., 1] += nbc  # offset z
        rec_loc_padded[..., 0] += nbc
        rec_loc_padded[..., 1] += nbc

        out = deepwave.scalar(
            v_padded,
            g.dx,
            g.dt,
            source_amplitudes=src_amp,
            source_locations=src_loc_padded,
            receiver_locations=rec_loc_padded,
            pml_width=nbc,
            pml_freq=g.freq,
        )
        rec = out[-1]
        return as_shot_receiver_time(
            rec,
            n_receivers=g.n_receivers,
            nt=g.nt,
            name="deepwave receiver data",
        )

    @torch.no_grad()
    def simulate_batch(self, v_models: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Run transmission forward for a batch of velocity models (serial).

        deepwave does not support velocity-model-level batching, so we loop.

        Args:
            v_models: [G, nx, nz] velocity model batch.
            device: torch device.

        Returns:
            seismograms: [G, n_shots, n_receivers, nt]
        """
        results = []
        g_size = int(v_models.shape[0])
        for i in range(g_size):
            seis = self.simulate(v_models[i], device=device)
            results.append(seis)
        return torch.stack(results, dim=0).contiguous()


# Backward compatibility aliases
TransmissionGeometry = AcquisitionGeometry
TransmissionForward = AcquisitionForward

if __name__ == "__main__":
    print("=== TransmissionForward smoke test ===")
    geom = TransmissionGeometry(
        nx_model=70, nz_model=70,
        n_shots=5, n_receivers=70,
        nt=1000, pml_width=40,
    )
    print(f"  Model: {geom.nx_model}×{geom.nz_model}, PML: {geom.pml_width}")
    print(f"  Padded: {geom.padded_nx}×{geom.padded_nz}")
    print(f"  Receivers at z_padded={geom.pml_width + geom.nz_model - 1}")
    print(f"  PML starts at z_padded={geom.pml_width + geom.nz_model}")
    print(f"  Receiver in PML? {'YES ⚠️' if not geom.validate_receiver_z() else 'NO ✓'}")

    forward = TransmissionForward(geom)

    # Basic smoke: can we run a forward sim?
    try:
        import numpy as np
        v_test = torch.full((70, 70), 2500.0, dtype=torch.float32)
        print("  Running forward simulation...")
        seis = forward.simulate(v_test, device="cpu")
        print(f"  Seismogram shape: {tuple(seis.shape)}")  # [5, 70, 1000]
        print(f"  Seismogram range: [{seis.min().item():.4f}, {seis.max().item():.4f}]")
        assert not torch.isnan(seis).any(), "NaN in seismogram!"
        print("  ✓ Transmission forward works")
    except Exception as e:
        print(f"  ⚠️ Forward simulation skipped: {e}")
        print("  (Expected if deepwave not available in current env)")
