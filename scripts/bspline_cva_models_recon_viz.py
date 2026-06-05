import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.Bspline import bspline2d_inverse, bspline2d_prolong


@dataclass(frozen=True)
class SampleRef:
    file: str
    index: int


def _key_numeric(p: str) -> Tuple[int, str]:
    stem = Path(p).stem
    digits = "".join([c for c in stem if c.isdigit()])
    return (int(digits) if digits else 0, p)


def iter_cva_models(
    model_dir: str,
    max_samples: int,
    start: int,
    stride: int,
) -> Iterable[Tuple[SampleRef, np.ndarray]]:
    files = [str(p) for p in Path(model_dir).glob("model*.npy")]
    files = sorted(files, key=_key_numeric)

    seen = 0
    skipped = 0
    for fp in files:
        arr = np.load(fp)
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected array shape in {fp}: {arr.shape}")

        n = arr.shape[0]
        for i in range(0, n, stride):
            if skipped < start:
                skipped += 1
                continue
            ref = SampleRef(file=Path(fp).name, index=i)
            yield ref, arr[i]
            seen += 1
            if seen >= max_samples:
                return


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
) -> torch.Tensor:
    c_hat = bspline2d_inverse(m_true, ctrl_shape, lam=lam, maxit=maxit, tol=tol, verbose=False)
    m_rec = bspline2d_prolong(c_hat, tuple(m_true.shape))
    return m_rec


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join("data", "CVA", "CurveVel_A", "model"),
    )
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--ctrlH", type=int, default=4)
    parser.add_argument("--ctrlW", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "bspline_cva_4x4_recon"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float32

    refs: List[SampleRef] = []
    models: List[torch.Tensor] = []
    for ref, m in iter_cva_models(args.model_dir, args.num, args.start, args.stride):
        refs.append(ref)
        models.append(torch.from_numpy(m).to(device=device, dtype=dtype))

    if not models:
        raise RuntimeError("No models loaded. Check --model_dir/--start/--stride/--num.")

    v_all = torch.stack(models, dim=0)
    vmin, vmax = v_all.min().item(), v_all.max().item()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for ref, m_true in zip(refs, models):
        m_rec = run_recon(m_true, (args.ctrlH, args.ctrlW), args.lam, args.maxit, args.tol)
        err = m_rec - m_true

        rmse = _rmse(m_true, m_rec)
        rel = _rel_l2(m_true, m_rec)
        mae = torch.mean(torch.abs(err)).item()
        emax = torch.max(torch.abs(err)).item()

        sid = f"{ref.file}:{ref.index}"
        rows.append((sid, rmse, rel, mae, emax))

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        im0 = ax[0].imshow(m_true.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[0].set_title(f"{sid} true")
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

        fig_path = os.path.join(args.out_dir, f"{ref.file}_idx{ref.index:04d}.png")
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)

    import csv

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "rmse", "rel_l2", "mae", "max_abs"])
        for r in rows:
            w.writerow(list(r))

    n = len(models)
    fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n), constrained_layout=True)
    if n == 1:
        ax = ax.reshape(1, 3)

    for i, (ref, m_true) in enumerate(zip(refs, models)):
        m_rec = run_recon(m_true, (args.ctrlH, args.ctrlW), args.lam, args.maxit, args.tol)
        err = m_rec - m_true
        eabs = float(torch.max(torch.abs(err)).item())
        sid = f"{ref.file}:{ref.index}"

        ax[i, 0].imshow(m_true.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[i, 0].set_title(f"{sid} true")
        ax[i, 1].imshow(m_rec.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[i, 1].set_title("recon")
        ax[i, 2].imshow(err.detach().cpu().numpy(), cmap="RdBu_r", vmin=-eabs, vmax=eabs)
        ax[i, 2].set_title("error")
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    fig_path = os.path.join(args.out_dir, "grid.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    rmse_all = np.array([r[1] for r in rows], dtype=np.float64)
    rel_all = np.array([r[2] for r in rows], dtype=np.float64)
    mae_all = np.array([r[3] for r in rows], dtype=np.float64)
    mx_all = np.array([r[4] for r in rows], dtype=np.float64)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_samples: {n}\n")
        f.write(f"vmin: {vmin:.3f}, vmax: {vmax:.3f}\n")
        f.write(f"rmse_mean: {rmse_all.mean():.6f}, rmse_std: {rmse_all.std():.6f}\n")
        f.write(f"rel_l2_mean: {rel_all.mean():.6f}, rel_l2_std: {rel_all.std():.6f}\n")
        f.write(f"mae_mean: {mae_all.mean():.6f}, mae_std: {mae_all.std():.6f}\n")
        f.write(f"max_abs_mean: {mx_all.mean():.6f}, max_abs_max: {mx_all.max():.6f}\n")

    print(f"Saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

