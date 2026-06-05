import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.Bspline import bspline2d_inverse, bspline2d_prolong


@dataclass(frozen=True)
class ModelCase:
    name: str
    m_true: torch.Tensor


def _mesh(H: int, W: int, device, dtype):
    z = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, W).expand(H, W)
    return z, x


def build_velocity_cases(H: int, W: int, device, dtype) -> List[ModelCase]:
    z, x = _mesh(H, W, device, dtype)
    cases: List[ModelCase] = []

    vmin = torch.tensor(1500.0, device=device, dtype=dtype)
    vmax = torch.tensor(4500.0, device=device, dtype=dtype)

    m_grad = vmin + (vmax - vmin) * z
    cases.append(ModelCase("linear_grad", m_grad))

    z0 = torch.tensor(0.55, device=device, dtype=dtype)
    sharp = torch.tensor(30.0, device=device, dtype=dtype)
    s = torch.sigmoid((z - z0) * sharp)
    m_layer = (1 - s) * 1800.0 + s * 3800.0
    cases.append(ModelCase("two_layer", m_layer))

    cx, cz = torch.tensor(0.62, device=device, dtype=dtype), torch.tensor(0.6, device=device, dtype=dtype)
    sigx, sigz = torch.tensor(0.10, device=device, dtype=dtype), torch.tensor(0.12, device=device, dtype=dtype)
    g = torch.exp(-0.5 * ((x - cx) / sigx) ** 2 - 0.5 * ((z - cz) / sigz) ** 2)
    m_blob = m_grad - 650.0 * g
    cases.append(ModelCase("gaussian_low_blob", m_blob.clamp(vmin, vmax)))

    kx = torch.tensor(8.0, device=device, dtype=dtype)
    kz = torch.tensor(8.0, device=device, dtype=dtype)
    patt = torch.sin(2 * torch.pi * kx * x) * torch.sin(2 * torch.pi * kz * z)
    m_checker = 3000.0 + 350.0 * patt
    cases.append(ModelCase("checkerboard", m_checker))

    x0 = torch.tensor(0.52, device=device, dtype=dtype)
    dz = torch.tensor(0.08, device=device, dtype=dtype)
    z_shift = z + dz * (x > x0).to(dtype)
    s2 = torch.sigmoid((z_shift - z0) * sharp)
    m_fault = (1 - s2) * 1900.0 + s2 * 3950.0
    cases.append(ModelCase("faulted_layers", m_fault))

    return cases


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm(a - b)
    den = torch.linalg.vector_norm(a).clamp_min(1e-12)
    return (num / den).item()


def run_recon(
    m_true: torch.Tensor,
    ctrl_shape: Tuple[int, int],
    lam: float,
    maxit: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_hat = bspline2d_inverse(m_true, ctrl_shape, lam=lam, maxit=maxit, tol=tol, verbose=False)
    m_rec = bspline2d_prolong(c_hat, tuple(m_true.shape))
    return c_hat, m_rec


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=70)
    parser.add_argument("--W", type=int, default=70)
    parser.add_argument("--ctrlH", type=int, default=4)
    parser.add_argument("--ctrlW", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "bspline_4x4_recon"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    os.makedirs(args.out_dir, exist_ok=True)

    cases = build_velocity_cases(args.H, args.W, device, dtype)
    v_all = torch.stack([c.m_true for c in cases], dim=0)
    vmin, vmax = v_all.min().item(), v_all.max().item()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for case in cases:
        _, m_rec = run_recon(case.m_true, (args.ctrlH, args.ctrlW), args.lam, args.maxit, args.tol)
        err = m_rec - case.m_true

        rmse = _rmse(case.m_true, m_rec)
        rel = _rel_l2(case.m_true, m_rec)
        mae = torch.mean(torch.abs(err)).item()
        emax = torch.max(torch.abs(err)).item()

        rows.append((case.name, rmse, rel, mae, emax))

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        im0 = ax[0].imshow(case.m_true.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[0].set_title(f"{case.name}: true")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(m_rec.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[1].set_title(f"recon (rmse={rmse:.1f}, rel={rel:.3f})")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        err_np = err.detach().cpu().numpy()
        eabs = float(torch.max(torch.abs(err)).item())
        im2 = ax[2].imshow(err_np, cmap="RdBu_r", vmin=-eabs, vmax=eabs)
        ax[2].set_title(f"error (mae={mae:.1f}, max={emax:.1f})")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        fig_path = os.path.join(args.out_dir, f"{case.name}.png")
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)

    import csv

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "rmse", "rel_l2", "mae", "max_abs"])
        for r in rows:
            w.writerow(list(r))

    grid = len(cases)
    fig, ax = plt.subplots(grid, 3, figsize=(12, 4 * grid), constrained_layout=True)
    if grid == 1:
        ax = ax.reshape(1, 3)

    for i, case in enumerate(cases):
        img_true = case.m_true.detach().cpu().numpy()
        _, m_rec = run_recon(case.m_true, (args.ctrlH, args.ctrlW), args.lam, args.maxit, args.tol)
        err = (m_rec - case.m_true).detach().cpu().numpy()
        eabs = float(torch.max(torch.abs(m_rec - case.m_true)).item())

        ax[i, 0].imshow(img_true, cmap="viridis", vmin=vmin, vmax=vmax)
        ax[i, 0].set_title(f"{case.name} true")
        ax[i, 1].imshow(m_rec.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[i, 1].set_title("recon")
        ax[i, 2].imshow(err, cmap="RdBu_r", vmin=-eabs, vmax=eabs)
        ax[i, 2].set_title("error")
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    fig_path = os.path.join(args.out_dir, "all_cases.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"Saved figures to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
