import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


def _parse_ctrl_shapes(ctrl_shapes: str) -> List[Tuple[int, int]]:
    s = ctrl_shapes.strip()
    if not s:
        return []
    shapes: List[Tuple[int, int]] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "x" not in p:
            raise ValueError(f"Bad shape '{p}', expect like 8x16")
        a, b = p.split("x", 1)
        h = int(a.strip())
        w = int(b.strip())
        shapes.append((h, w))
    return shapes


def _ctrl_label(hw: Tuple[int, int]) -> str:
    return f"{hw[0]}x{hw[1]}"


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


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


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
    parser.add_argument("--ctrl_sizes", type=str, default="")
    parser.add_argument("--ctrl_shapes", type=str, default="")
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--maxit", type=int, default=160)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "bspline_cva_multi_ctrl"))
    args = parser.parse_args()

    ctrl_shapes: List[Tuple[int, int]] = []
    if args.ctrl_sizes.strip():
        ctrl_sizes: Sequence[int] = tuple(int(s.strip()) for s in args.ctrl_sizes.split(",") if s.strip())
        ctrl_shapes.extend([(s, s) for s in ctrl_sizes])
    ctrl_shapes.extend(_parse_ctrl_shapes(args.ctrl_shapes))
    if not ctrl_shapes:
        raise ValueError("Need at least one shape from --ctrl_sizes or --ctrl_shapes")
    ctrl_shapes = sorted(list(dict.fromkeys(ctrl_shapes)), key=lambda hw: (hw[0] * hw[1], hw[0], hw[1]))

    device = torch.device(args.device)
    dtype = torch.float32

    refs: List[SampleRef] = []
    models: List[torch.Tensor] = []
    for ref, m in iter_cva_models(args.model_dir, args.num, args.start, args.stride):
        refs.append(ref)
        models.append(torch.from_numpy(m).to(device=device, dtype=dtype))

    if not models:
        raise RuntimeError("No models loaded. Check --model_dir/--start/--stride/--num.")

    base_out = _ensure_dir(args.out_dir)

    v_all = torch.stack(models, dim=0)
    vmin, vmax = v_all.min().item(), v_all.max().item()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_by_ctrl: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
    sample_preview = min(6, len(models))

    for hw in ctrl_shapes:
        h, w = hw
        label = _ctrl_label(hw)
        out_cs = _ensure_dir(os.path.join(base_out, f"ctrl_{label}"))
        rows: List[Tuple[str, float, float, float, float]] = []
        for ref, m_true in zip(refs, models):
            m_rec = run_recon(m_true, (h, w), args.lam, args.maxit, args.tol)
            err = m_rec - m_true
            sid = f"{ref.file}:{ref.index}"
            rmse = _rmse(m_true, m_rec)
            rel = _rel_l2(m_true, m_rec)
            mae = torch.mean(torch.abs(err)).item()
            emax = torch.max(torch.abs(err)).item()
            rows.append((sid, rmse, rel, mae, emax))

        metrics_by_ctrl[label] = rows

        import csv

        csv_path = os.path.join(out_cs, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample", "rmse", "rel_l2", "mae", "max_abs"])
            for r in rows:
                w.writerow(list(r))

    out_preview = _ensure_dir(os.path.join(base_out, "previews"))
    for i in range(sample_preview):
        ref, m_true = refs[i], models[i]
        sid = f"{ref.file}:{ref.index}"
        fig, ax = plt.subplots(len(ctrl_shapes) + 1, 2, figsize=(10, 3.2 * (len(ctrl_shapes) + 1)), constrained_layout=True)
        if ax.ndim == 1:
            ax = ax.reshape(-1, 2)

        im0 = ax[0, 0].imshow(m_true.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax[0, 0].set_title(f"{sid} true")
        fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)
        ax[0, 1].axis("off")

        for r, hw in enumerate(ctrl_shapes, start=1):
            h, w = hw
            label = _ctrl_label(hw)
            m_rec = run_recon(m_true, (h, w), args.lam, args.maxit, args.tol)
            err = m_rec - m_true
            rmse = _rmse(m_true, m_rec)
            rel = _rel_l2(m_true, m_rec)

            im1 = ax[r, 0].imshow(m_rec.detach().cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            ax[r, 0].set_title(f"recon {label} (rmse={rmse:.1f}, rel={rel:.3f})")
            fig.colorbar(im1, ax=ax[r, 0], fraction=0.046, pad=0.04)

            eabs = float(torch.max(torch.abs(err)).item())
            im2 = ax[r, 1].imshow(err.detach().cpu().numpy(), cmap="RdBu_r", vmin=-eabs, vmax=eabs)
            ax[r, 1].set_title("error")
            fig.colorbar(im2, ax=ax[r, 1], fraction=0.046, pad=0.04)

        for a in ax.reshape(-1):
            a.set_xticks([])
            a.set_yticks([])

        fig_path = os.path.join(out_preview, f"{ref.file}_idx{ref.index:04d}_multi.png")
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)

    ctrl_sorted = [_ctrl_label(hw) for hw in ctrl_shapes]
    rmse_mean = []
    rmse_std = []
    rel_mean = []
    rel_std = []
    mae_mean = []
    mae_std = []
    mx_mean = []
    mx_max = []
    n_ctrl = []

    for label, hw in zip(ctrl_sorted, ctrl_shapes):
        rows = metrics_by_ctrl[label]
        rmse = np.array([r[1] for r in rows], dtype=np.float64)
        rel = np.array([r[2] for r in rows], dtype=np.float64)
        mae = np.array([r[3] for r in rows], dtype=np.float64)
        mx = np.array([r[4] for r in rows], dtype=np.float64)
        n_ctrl.append(hw[0] * hw[1])

        rmse_mean.append(rmse.mean())
        rmse_std.append(rmse.std())
        rel_mean.append(rel.mean())
        rel_std.append(rel.std())
        mae_mean.append(mae.mean())
        mae_std.append(mae.std())
        mx_mean.append(mx.mean())
        mx_max.append(mx.max())

    import csv

    summary_csv = os.path.join(base_out, "summary_by_ctrl.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ctrl", "n_ctrl", "rmse_mean", "rmse_std", "rel_l2_mean", "rel_l2_std", "mae_mean", "mae_std", "max_abs_mean", "max_abs_max"])
        for i, label in enumerate(ctrl_sorted):
            w.writerow([label, n_ctrl[i], rmse_mean[i], rmse_std[i], rel_mean[i], rel_std[i], mae_mean[i], mae_std[i], mx_mean[i], mx_max[i]])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].errorbar(n_ctrl, rel_mean, yerr=rel_std, marker="o", linewidth=2)
    ax[0].set_xlabel("num control points (H*W)")
    ax[0].set_ylabel("rel L2 (mean ± std)")
    ax[0].grid(True, alpha=0.3)
    for x, y, label in zip(n_ctrl, rel_mean, ctrl_sorted):
        ax[0].annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))

    ax[1].errorbar(n_ctrl, rmse_mean, yerr=rmse_std, marker="o", linewidth=2)
    ax[1].set_xlabel("num control points (H*W)")
    ax[1].set_ylabel("RMSE (mean ± std)")
    ax[1].grid(True, alpha=0.3)
    for x, y, label in zip(n_ctrl, rmse_mean, ctrl_sorted):
        ax[1].annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))

    fig_path = os.path.join(base_out, "ctrl_vs_error.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    txt_path = os.path.join(base_out, "run_info.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"num_samples: {len(models)}\n")
        f.write(f"ctrl_shapes: {','.join(ctrl_sorted)}\n")
        f.write(f"lam: {args.lam}\n")
        f.write(f"maxit: {args.maxit}\n")
        f.write(f"tol: {args.tol}\n")
        f.write(f"vmin: {vmin:.3f}, vmax: {vmax:.3f}\n")

    print(f"Saved to: {os.path.abspath(base_out)}")


if __name__ == "__main__":
    main()
