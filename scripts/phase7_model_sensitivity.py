from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_shape(text: str) -> tuple[int, int]:
    if "x" not in text.lower():
        raise ValueError(f"Shape must look like 30x10 for [nx_ctrl,nz_ctrl], got {text!r}")
    a, b = text.lower().split("x", 1)
    return int(a), int(b)


def ensure_2d_velocity(v: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"{name} must be 2D [nx,nz], got {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return v


def load_marmousi(path: Path, orig_xl: int, orig_zl: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    expected = int(orig_xl) * int(orig_zl)
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} floats, expected {expected} for {orig_xl}x{orig_zl}")
    # This Marmousi binary is already ordered as [x, z] after reshape(460, 150).
    # Returning [nx, nz] keeps the model aligned with Deepwave and B-spline code;
    # plotting transposes it once to display [z, x].
    return raw.reshape(int(orig_zl), int(orig_xl)).astype(np.float32)


def infer_salt_nx(path: Path, salt_nz: int, salt_nx: int | None) -> int:
    count = path.stat().st_size // np.dtype(np.float32).itemsize
    if salt_nx is not None:
        expected = int(salt_nz) * int(salt_nx)
        if count != expected:
            raise ValueError(
                f"{path.name} has {count} floats, but salt_nz*salt_nx={expected}. "
                f"For salt_nz={salt_nz}, inferred salt_nx would be {count // salt_nz} if divisible."
            )
        return int(salt_nx)
    if count % int(salt_nz) != 0:
        raise ValueError(f"Cannot infer salt_nx: {count} floats is not divisible by salt_nz={salt_nz}")
    return int(count // int(salt_nz))


def load_salt(path: Path, salt_nz: int, salt_nx: int | None) -> np.ndarray:
    nx = infer_salt_nx(path, salt_nz=salt_nz, salt_nx=salt_nx)
    raw = np.fromfile(path, dtype=np.float32)
    # The salt-body binary is MATLAB/Fortran ordered: z changes smoothly down
    # each x-column. Return [nx,nz] for the project convention.
    return raw.reshape((int(salt_nz), int(nx)), order="F").T.astype(np.float32)


def synthetic_transmission_model(nx: int, nz: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, int(nx), dtype=np.float32)[:, None]
    z = np.linspace(0.0, 1.0, int(nz), dtype=np.float32)[None, :]

    background = 1700.0 + 2100.0 * z
    dipping_center = 0.30 + 0.34 * z
    salt_curtain = np.exp(-0.5 * ((x - dipping_center) / 0.075) ** 2)
    low_channel = np.exp(-0.5 * ((x - (0.72 - 0.18 * z)) / 0.10) ** 2 - 0.5 * ((z - 0.48) / 0.24) ** 2)
    shallow_cap = np.exp(-0.5 * ((z - 0.20) / 0.055) ** 2)

    model = background + 1350.0 * salt_curtain - 780.0 * low_channel + 280.0 * shallow_cap
    return np.clip(model, 1500.0, 4700.0).astype(np.float32)


def crop_model(v: np.ndarray, x0: int, z0: int, crop_nx: int | None, crop_nz: int | None) -> np.ndarray:
    nx, nz = v.shape
    x0 = int(x0)
    z0 = int(z0)
    if x0 < 0 or z0 < 0 or x0 >= nx or z0 >= nz:
        raise ValueError(f"Crop origin ({x0},{z0}) outside model shape {v.shape}")
    x1 = nx if crop_nx is None or crop_nx <= 0 else min(nx, x0 + int(crop_nx))
    z1 = nz if crop_nz is None or crop_nz <= 0 else min(nz, z0 + int(crop_nz))
    cropped = v[x0:x1, z0:z1]
    if cropped.size == 0:
        raise ValueError(f"Empty crop from shape {v.shape}, origin=({x0},{z0}), end=({x1},{z1})")
    return cropped.astype(np.float32)


def resize_model(v: np.ndarray, target_nx: int, target_nz: int) -> np.ndarray:
    v = ensure_2d_velocity(v, "model").astype(np.float64)
    nx, nz = v.shape
    x_old = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    z_old = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, int(target_nx), dtype=np.float64)
    z_new = np.linspace(0.0, 1.0, int(target_nz), dtype=np.float64)

    tmp = np.empty((int(target_nx), nz), dtype=np.float64)
    for iz in range(nz):
        tmp[:, iz] = np.interp(x_new, x_old, v[:, iz])

    out = np.empty((int(target_nx), int(target_nz)), dtype=np.float64)
    for ix in range(int(target_nx)):
        out[ix, :] = np.interp(z_new, z_old, tmp[ix, :])
    return out.astype(np.float32)


def bspline_roundtrip(model: np.ndarray, ctrl_shape: tuple[int, int], lam: float, maxit: int, device: str):
    import torch

    from agents.Bspline import bspline2d_inverse, bspline2d_prolong

    torch_device = torch.device(device)
    m = torch.from_numpy(model.astype(np.float32)).to(torch_device)
    ctrl = bspline2d_inverse(m, ctrl_shape, lam=float(lam), maxit=int(maxit), tol=1e-12, verbose=False)
    smooth = bspline2d_prolong(ctrl, tuple(m.shape)).contiguous()
    return ctrl.detach().cpu().numpy().astype(np.float32), smooth.detach().cpu().numpy().astype(np.float32)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def simulate_deepwave(
    velocity: np.ndarray,
    *,
    dx: float,
    dt: float,
    freq: float,
    nt: int,
    n_shots: int,
    n_receivers: int,
    pml_width: int,
    source_depth: str,
    receiver_depth: str,
    device: str,
) -> np.ndarray:
    import torch

    from agents.transmission_forward import AcquisitionForward, AcquisitionGeometry

    device = resolve_device(device)
    geom = AcquisitionGeometry(
        nx_model=int(velocity.shape[0]),
        nz_model=int(velocity.shape[1]),
        dx=float(dx),
        dt=float(dt),
        freq=float(freq),
        nt=int(nt),
        n_shots=int(n_shots),
        n_receivers=int(n_receivers),
        pml_width=int(pml_width),
        geometry="transmission",
        source_depth=str(source_depth),
        receiver_depth=str(receiver_depth),
    )
    forward = AcquisitionForward(geom)
    v_t = torch.from_numpy(np.asarray(velocity, dtype=np.float32))
    gather = forward.simulate(v_t, device=device).detach().cpu().numpy().astype(np.float32)
    if gather.shape == (n_shots, nt, n_receivers):
        gather = np.transpose(gather, (0, 2, 1))
    if gather.shape != (n_shots, n_receivers, nt):
        raise ValueError(f"Expected gather [shot,receiver,time]={(n_shots, n_receivers, nt)}, got {gather.shape}")
    return gather


def uniform_velocity_value(v_true: np.ndarray, value: str) -> float:
    value = str(value).strip().lower()
    if value == "mean":
        return float(np.mean(v_true))
    if value == "median":
        return float(np.median(v_true))
    return float(value)


def normalize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    peak = float(max(np.max(np.abs(a)), np.max(np.abs(b))))
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0
    return (a / peak).astype(np.float32), (b / peak).astype(np.float32), peak


def gather_metrics(true_g: np.ndarray, uniform_g: np.ndarray) -> dict[str, float]:
    diff = true_g - uniform_g
    rmse = float(np.sqrt(np.mean(diff**2)))
    rel = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(true_g.ravel()), 1e-12))
    max_abs = float(np.max(np.abs(diff)))
    dot = float(np.dot(true_g.ravel(), uniform_g.ravel()))
    den = float(np.linalg.norm(true_g.ravel()) * np.linalg.norm(uniform_g.ravel()))
    corr = dot / den if den > 0 else 0.0
    return {"waveform_rmse_norm": rmse, "waveform_rel_l2": rel, "waveform_max_abs_norm": max_abs, "waveform_corr": corr}


def robust_abs_limit(data: np.ndarray, percentile: float = 99.0) -> float:
    limit = float(np.percentile(np.abs(data), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return limit


def plot_velocity_panel(base: np.ndarray, v_true: np.ndarray, uniform: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    all_v = np.stack([base, v_true, uniform])
    vmin, vmax = float(all_v.min()), float(all_v.max())
    items = [
        ("cropped/resampled input", base, "turbo", vmin, vmax),
        ("B-spline v_true", v_true, "turbo", vmin, vmax),
        ("uniform model", uniform, "turbo", vmin, vmax),
        ("v_true - uniform", v_true - uniform, "RdBu_r", None, None),
    ]
    fig.suptitle(title, fontsize=13)
    for ax, (label, arr, cmap, lo, hi) in zip(axes, items):
        if lo is None:
            e = float(np.max(np.abs(arr)))
            lo, hi = -e, e
        im = ax.imshow(arr.T, origin="upper", cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_single_wiggle(ax: plt.Axes, shot: np.ndarray, *, color: str, dt: float, dx: float, scale: float) -> None:
    n_receivers, nt = shot.shape
    time = np.arange(nt, dtype=np.float32) * float(dt)
    receiver_km = np.arange(n_receivers, dtype=np.float32) * float(dx) / 1000.0
    spacing = float(dx) / 1000.0
    peak = np.max(np.abs(shot), axis=1, keepdims=True)
    normalized = shot / np.where(peak > 1e-12, peak, 1.0)
    for i in range(n_receivers):
        x0 = receiver_km[i]
        ax.plot(np.full_like(time, x0), time, color="0.88", linewidth=0.15, alpha=0.35)
        ax.plot(x0 + normalized[i] * spacing * scale, time, color=color, linewidth=0.38, alpha=0.78)


def plot_wiggle_overlay(true_g: np.ndarray, uniform_g: np.ndarray, out_path: Path, *, dt: float, dx: float) -> None:
    n_shots, n_receivers, nt = true_g.shape
    receiver_max = (n_receivers - 1) * float(dx) / 1000.0
    time_max = (nt - 1) * float(dt)
    fig = plt.figure(figsize=(16.0, max(4.0, 2.55 * n_shots)), constrained_layout=True)
    gs = fig.add_gridspec(n_shots, 2, width_ratios=[1.0, 1.0])
    fig.suptitle("Phase7 sensitivity: v_true vs uniform velocity", fontsize=14)

    diff_lim = robust_abs_limit(true_g - uniform_g)
    for shot_i in range(n_shots):
        ax_img = fig.add_subplot(gs[shot_i, 0])
        im = ax_img.imshow(
            (true_g[shot_i] - uniform_g[shot_i]).T,
            cmap="seismic",
            origin="upper",
            aspect="auto",
            vmin=-diff_lim,
            vmax=diff_lim,
            extent=[0, receiver_max, time_max, 0],
        )
        ax_img.set_title(f"Shot {shot_i}: normalized gather difference")
        ax_img.set_xlabel("Receiver x (km)")
        ax_img.set_ylabel("Time (s)")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        ax_wig = fig.add_subplot(gs[shot_i, 1])
        plot_single_wiggle(ax_wig, true_g[shot_i], color="black", dt=dt, dx=dx, scale=0.45)
        plot_single_wiggle(ax_wig, uniform_g[shot_i], color="#dc2626", dt=dt, dx=dx, scale=0.45)
        ax_wig.set_title(f"Shot {shot_i}: wiggle overlay")
        ax_wig.set_xlim(-float(dx) / 1000.0, receiver_max + float(dx) / 1000.0)
        ax_wig.set_ylim(time_max, 0)
        ax_wig.set_xlabel("Receiver x (km)")
        ax_wig.set_ylabel("Time (s)")
        ax_wig.grid(True, color="0.9", linewidth=0.5)
        if shot_i == 0:
            ax_wig.plot([], [], color="black", label="v_true")
            ax_wig.plot([], [], color="#dc2626", label="uniform")
            ax_wig.legend(loc="lower right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_base_model(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    models_dir = PROJECT_ROOT / "models"
    meta: dict[str, object] = {"model": args.model}
    if args.model == "marmousi":
        raw = load_marmousi(models_dir / args.marmousi_file, args.marmousi_orig_xl, args.marmousi_orig_zl)
        meta.update({"raw_shape_nx_nz": list(raw.shape), "source_file": args.marmousi_file})
    elif args.model == "salt":
        raw = load_salt(models_dir / args.salt_file, args.salt_nz, args.salt_nx)
        meta.update({"raw_shape_nx_nz": list(raw.shape), "source_file": args.salt_file, "dx_original": args.salt_dx, "dz_original": args.salt_dz})
    elif args.model == "synthetic":
        raw = synthetic_transmission_model(args.target_nx, args.target_nz)
        meta.update({"raw_shape_nx_nz": list(raw.shape), "source_file": "procedural"})
    else:
        raise ValueError(f"Unknown model {args.model!r}")

    if args.model == "synthetic":
        cropped = raw
    else:
        cropped = crop_model(raw, args.crop_x0, args.crop_z0, args.crop_nx, args.crop_nz)
    base = resize_model(cropped, args.target_nx, args.target_nz)
    meta.update({"crop_shape_nx_nz": list(cropped.shape), "target_shape_nx_nz": [args.target_nx, args.target_nz]})
    return base, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase7 models and compare v_true vs uniform transmission gathers.")
    parser.add_argument("--model", choices=["marmousi", "salt", "synthetic"], default="marmousi")
    parser.add_argument("--marmousi_file", default="marmousi_v.bin")
    parser.add_argument("--marmousi_orig_xl", type=int, default=150)
    parser.add_argument("--marmousi_orig_zl", type=int, default=460)
    parser.add_argument("--salt_file", default="vel_z6.25m_x12.5m_exact.bin")
    parser.add_argument("--salt_nz", type=int, default=1911)
    parser.add_argument("--salt_nx", type=int, default=None)
    parser.add_argument("--salt_dx", type=float, default=12.5)
    parser.add_argument("--salt_dz", type=float, default=6.25)
    parser.add_argument("--crop_x0", type=int, default=0)
    parser.add_argument("--crop_z0", type=int, default=0)
    parser.add_argument("--crop_nx", type=int, default=0)
    parser.add_argument("--crop_nz", type=int, default=0)
    parser.add_argument("--target_nx", type=int, default=210)
    parser.add_argument("--target_nz", type=int, default=70)
    parser.add_argument("--ctrl_shape", default="30x10", help="B-spline control grid as nx_ctrl x nz_ctrl.")
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--maxit", type=int, default=180)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--freq", type=float, default=15.0)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--n_shots", type=int, default=7)
    parser.add_argument("--n_receivers", type=int, default=0, help="0 means target_nx receivers.")
    parser.add_argument("--pml_width", type=int, default=40)
    parser.add_argument("--source_depth", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--receiver_depth", choices=["top", "bottom"], default="top")
    parser.add_argument("--uniform_velocity", default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_forward", action="store_true")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    ctrl_shape = parse_shape(args.ctrl_shape)
    n_receivers = int(args.target_nx if args.n_receivers <= 0 else args.n_receivers)
    device = resolve_device(args.device)

    base, meta = build_base_model(args)
    ctrl, v_true = bspline_roundtrip(base, ctrl_shape=ctrl_shape, lam=args.lam, maxit=args.maxit, device=device)
    uniform_value = uniform_velocity_value(v_true, args.uniform_velocity)
    uniform = np.full_like(v_true, uniform_value, dtype=np.float32)

    label = f"{args.model}_nx{args.target_nx}_nz{args.target_nz}_ctrl{ctrl_shape[0]}x{ctrl_shape[1]}"
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "outputs" / "phase7_sensitivity" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "base_model.npy", base)
    np.save(out_dir / "bspline_control.npy", ctrl)
    np.save(out_dir / "v_true.npy", v_true)
    np.save(out_dir / "uniform_velocity.npy", uniform)
    plot_velocity_panel(base, v_true, uniform, out_dir / "velocity_models.png", title=label)

    run_meta = {
        **meta,
        "array_convention": "saved velocity arrays are [nx,nz]; plots display velocity.T as [z,x]",
        "ctrl_shape": list(ctrl_shape),
        "lam": args.lam,
        "maxit": args.maxit,
        "uniform_velocity": uniform_value,
        "dx": args.dx,
        "dt": args.dt,
        "freq": args.freq,
        "nt": args.nt,
        "n_shots": args.n_shots,
        "n_receivers": n_receivers,
        "pml_width": args.pml_width,
        "source_depth": args.source_depth,
        "receiver_depth": args.receiver_depth,
        "device": device,
    }

    if not args.skip_forward:
        true_g = simulate_deepwave(
            v_true,
            dx=args.dx,
            dt=args.dt,
            freq=args.freq,
            nt=args.nt,
            n_shots=args.n_shots,
            n_receivers=n_receivers,
            pml_width=args.pml_width,
            source_depth=args.source_depth,
            receiver_depth=args.receiver_depth,
            device=device,
        )
        uniform_g = simulate_deepwave(
            uniform,
            dx=args.dx,
            dt=args.dt,
            freq=args.freq,
            nt=args.nt,
            n_shots=args.n_shots,
            n_receivers=n_receivers,
            pml_width=args.pml_width,
            source_depth=args.source_depth,
            receiver_depth=args.receiver_depth,
            device=device,
        )
        true_g, uniform_g, scale = normalize_pair(true_g, uniform_g)
        np.save(out_dir / "gather_v_true_norm.npy", true_g)
        np.save(out_dir / "gather_uniform_norm.npy", uniform_g)
        plot_wiggle_overlay(true_g, uniform_g, out_dir / "wiggle_v_true_vs_uniform.png", dt=args.dt, dx=args.dx)
        run_meta.update({"normalization_peak": scale, **gather_metrics(true_g, uniform_g)})

    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Saved Phase7 sensitivity outputs to: {out_dir.resolve()}")
    if "waveform_rel_l2" in run_meta:
        print(
            "Sensitivity metrics: "
            f"rel_l2={run_meta['waveform_rel_l2']:.4f}, "
            f"rmse={run_meta['waveform_rmse_norm']:.4f}, "
            f"corr={run_meta['waveform_corr']:.4f}"
        )


if __name__ == "__main__":
    main()
