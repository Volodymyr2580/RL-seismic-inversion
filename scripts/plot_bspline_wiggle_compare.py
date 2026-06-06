from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def natural_key(path: Path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else path.stem


def reflect_indices(idx: np.ndarray, n: int) -> np.ndarray:
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * (n - 1)
    idx_mod = np.mod(idx, period)
    return np.where(idx_mod <= n - 1, idx_mod, period - idx_mod)


def cubic_weights(t: np.ndarray) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return np.stack(
        [
            (1 - t) ** 3 / 6.0,
            (3 * t3 - 6 * t2 + 4) / 6.0,
            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
            t3 / 6.0,
        ],
        axis=-1,
    )


def bspline_basis_1d(n_ctrl: int, n_dense: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n_dense, dtype=np.float64)
    base = s * (n_ctrl - 1)
    i1 = np.floor(base).astype(np.int64)
    t = base - i1
    idx = np.stack([i1 - 1, i1, i1 + 1, i1 + 2], axis=-1)
    idx = reflect_indices(idx, n_ctrl)
    weights = cubic_weights(t)

    basis = np.zeros((n_dense, n_ctrl), dtype=np.float64)
    rows = np.arange(n_dense)
    for k in range(4):
        np.add.at(basis, (rows, idx[:, k]), weights[:, k])
    return basis


def bspline_smooth_model(model: np.ndarray, ctrl_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h_ctrl, w_ctrl = ctrl_shape
    h_dense, w_dense = model.shape
    bz = bspline_basis_1d(h_ctrl, h_dense)
    bx = bspline_basis_1d(w_ctrl, w_dense)
    control = np.linalg.pinv(bz) @ model.astype(np.float64) @ np.linalg.pinv(bx.T)
    smooth = bz @ control @ bx.T
    return control.astype(np.float32), smooth.astype(np.float32)


def load_cva_model(root: str, file_idx: int, sample_idx: int) -> tuple[np.ndarray, dict]:
    root_path = Path(root)
    model_files = sorted((root_path / "model").glob("model*.npy"), key=natural_key)
    if not model_files:
        raise FileNotFoundError(f"No model*.npy files found under {root_path / 'model'}")
    if file_idx < 0 or file_idx >= len(model_files):
        raise ValueError(f"file_idx={file_idx} outside [0, {len(model_files) - 1}]")
    data = np.load(model_files[file_idx], mmap_mode="r")
    if sample_idx < 0 or sample_idx >= data.shape[0]:
        raise ValueError(f"sample_idx={sample_idx} outside [0, {data.shape[0] - 1}]")
    model = np.asarray(data[sample_idx], dtype=np.float32)
    if model.ndim == 3 and model.shape[0] == 1:
        model = model[0]
    return model, {"model_file": model_files[file_idx].name, "file_idx": file_idx, "sample_idx": sample_idx}


def load_velocity(path: str) -> np.ndarray:
    velocity = np.load(path).astype(np.float32)
    if velocity.ndim == 3 and velocity.shape[0] == 1:
        velocity = velocity[0]
    if velocity.shape != (70, 70):
        raise ValueError(f"Expected velocity shape (70, 70), got {velocity.shape}: {path}")
    return velocity


def ricker(freq: float, nt: int, dt: float) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * dt
    peak_time = 1.5 / freq
    arg = np.pi * freq * (t - peak_time)
    return (1.0 - 2.0 * arg * arg) * np.exp(-(arg * arg))


def damping_mask(nx: int, nz: int, width: int = 12, strength: float = 0.015) -> np.ndarray:
    mask = np.ones((nx, nz), dtype=np.float32)
    for i in range(nx):
        dx = min(i, nx - 1 - i)
        for j in range(nz):
            dz = min(j, nz - 1 - j)
            dist = min(dx, dz)
            if dist < width:
                x = (width - dist) / width
                mask[i, j] = np.exp(-strength * x * x)
    return mask


def simulate_fd(
    velocity: np.ndarray,
    geometry: str,
    n_shots: int,
    n_receivers: int,
    nt: int,
    dx: float,
    dt: float,
    freq: float,
) -> np.ndarray:
    """Small NumPy acoustic finite-difference simulator for visualization."""
    nx, nz = velocity.shape
    cfl2 = (velocity.astype(np.float32) * dt / dx) ** 2
    wavelet = ricker(freq, nt, dt)
    damp = damping_mask(nx, nz)
    src_xs = np.rint(np.linspace(0, nx - 1, n_shots)).astype(np.int64)
    rec_xs = np.rint(np.linspace(0, nx - 1, n_receivers)).astype(np.int64)
    src_z = 1
    rec_z = 1 if geometry == "reflection" else nz - 2

    gathers = np.zeros((n_shots, n_receivers, nt), dtype=np.float32)
    lap = np.zeros((nx, nz), dtype=np.float32)
    source_gain = 80.0

    for shot_i, src_x in enumerate(src_xs):
        prev = np.zeros((nx, nz), dtype=np.float32)
        curr = np.zeros((nx, nz), dtype=np.float32)
        for it in range(nt):
            lap.fill(0.0)
            lap[1:-1, 1:-1] = (
                curr[2:, 1:-1]
                + curr[:-2, 1:-1]
                + curr[1:-1, 2:]
                + curr[1:-1, :-2]
                - 4.0 * curr[1:-1, 1:-1]
            )
            nxt = 2.0 * curr - prev + cfl2 * lap
            nxt[src_x, src_z] += source_gain * wavelet[it]
            nxt *= damp
            curr *= damp
            gathers[shot_i, :, it] = curr[rec_xs, rec_z]
            prev, curr = curr, nxt

    peak = float(np.max(np.abs(gathers)))
    if peak > 0:
        gathers /= peak
    return gathers


def simulate_deepwave(
    velocity: np.ndarray,
    geometry: str,
    n_shots: int,
    n_receivers: int,
    nt: int,
    dx: float,
    dt: float,
    freq: float,
    pml_width: int,
    device: str,
) -> np.ndarray:
    import torch

    from agents.transmission_forward import AcquisitionForward, AcquisitionGeometry

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        geometry=str(geometry),
    )
    forward = AcquisitionForward(geom)
    v_t = torch.from_numpy(np.asarray(velocity, dtype=np.float32))
    gathers = forward.simulate(v_t, device=device).detach().cpu().numpy().astype(np.float32)
    if gathers.shape[1] == nt and gathers.shape[2] == n_receivers:
        gathers = np.transpose(gathers, (0, 2, 1))
    if gathers.shape != (n_shots, n_receivers, nt):
        raise ValueError(
            "Deepwave gather shape must be [shot, receiver, time], "
            f"got {gathers.shape}, expected {(n_shots, n_receivers, nt)}"
        )
    peak = float(np.max(np.abs(gathers)))
    if peak > 0:
        gathers /= peak
    return gathers


def simulate_gathers(
    velocity: np.ndarray,
    simulator: str,
    geometry: str,
    n_shots: int,
    n_receivers: int,
    nt: int,
    dx: float,
    dt: float,
    freq: float,
    pml_width: int,
    device: str,
) -> np.ndarray:
    if simulator == "deepwave":
        return simulate_deepwave(
            velocity,
            geometry=geometry,
            n_shots=n_shots,
            n_receivers=n_receivers,
            nt=nt,
            dx=dx,
            dt=dt,
            freq=freq,
            pml_width=pml_width,
            device=device,
        )
    return simulate_fd(
        velocity,
        geometry=geometry,
        n_shots=n_shots,
        n_receivers=n_receivers,
        nt=nt,
        dx=dx,
        dt=dt,
        freq=freq,
    )


def robust_clip(data: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    limit = float(np.percentile(np.abs(data), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def plot_single_wiggle(ax: plt.Axes, shot: np.ndarray, color: str, dt: float, dx: float, scale: float) -> None:
    n_receivers, nt = shot.shape
    time = np.arange(nt, dtype=np.float32) * dt
    receiver_km = np.arange(n_receivers, dtype=np.float32) * dx / 1000.0
    spacing = dx / 1000.0
    peak = np.max(np.abs(shot), axis=1, keepdims=True)
    normalized = shot / np.where(peak > 1e-12, peak, 1.0)
    for i in range(n_receivers):
        x0 = receiver_km[i]
        ax.plot(x0 + normalized[i] * spacing * scale, time, color=color, linewidth=0.45, alpha=0.82)


def plot_5shot_panel(
    model: np.ndarray,
    gathers: np.ndarray,
    title: str,
    out_path: Path,
    dt: float,
    dx: float,
    color: str,
) -> None:
    n_shots, n_receivers, nt = gathers.shape
    fig = plt.figure(figsize=(14.5, 14.0), constrained_layout=True)
    gs = fig.add_gridspec(n_shots, 3, width_ratios=[1.0, 1.15, 1.15])
    fig.suptitle(title, fontsize=14)

    receiver_max = (n_receivers - 1) * dx / 1000.0
    time_max = (nt - 1) * dt
    vmin_s, vmax_s = robust_clip(gathers)

    ax_model = fig.add_subplot(gs[:, 0])
    im_model = ax_model.imshow(
        model,
        cmap="turbo",
        origin="upper",
        aspect="equal",
        extent=[0, model.shape[1] * dx / 1000.0, model.shape[0] * dx / 1000.0, 0],
    )
    ax_model.set_title("B-spline smooth model")
    ax_model.set_xlabel("x (km)")
    ax_model.set_ylabel("z (km)")
    cbar = fig.colorbar(im_model, ax=ax_model, fraction=0.046, pad=0.04)
    cbar.set_label("m/s")

    for shot_i in range(n_shots):
        shot = gathers[shot_i]
        ax_img = fig.add_subplot(gs[shot_i, 1])
        im = ax_img.imshow(
            shot.T,
            cmap="seismic",
            origin="upper",
            aspect="auto",
            vmin=vmin_s,
            vmax=vmax_s,
            extent=[0, receiver_max, time_max, 0],
        )
        ax_img.set_title(f"Shot {shot_i} gather")
        ax_img.set_xlabel("Receiver x (km)")
        ax_img.set_ylabel("Time (s)")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        ax_wig = fig.add_subplot(gs[shot_i, 2])
        plot_single_wiggle(ax_wig, shot, color=color, dt=dt, dx=dx, scale=0.55)
        ax_wig.set_title(f"Shot {shot_i} wiggle")
        ax_wig.set_xlim(-dx / 1000.0, receiver_max + dx / 1000.0)
        ax_wig.set_ylim(time_max, 0)
        ax_wig.set_xlabel("Receiver x (km)")
        ax_wig.set_ylabel("Time (s)")
        ax_wig.grid(True, color="0.9", linewidth=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_compare_panel(
    models: list[np.ndarray],
    gathers: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    out_path: Path,
    dt: float,
    dx: float,
) -> None:
    n_shots, n_receivers, nt = gathers[0].shape
    fig = plt.figure(figsize=(16.0, 16.0), constrained_layout=True)
    gs = fig.add_gridspec(n_shots + 1, 3, height_ratios=[1.15] + [1.0] * n_shots)
    fig.suptitle("Velocity and shot-gather comparison", fontsize=14)

    all_v = np.stack(models)
    vmin, vmax = float(all_v.min()), float(all_v.max())
    for i, (model, label) in enumerate(zip(models, labels)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(
            model,
            cmap="turbo",
            origin="upper",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            extent=[0, model.shape[1] * dx / 1000.0, model.shape[0] * dx / 1000.0, 0],
        )
        ax.set_title(label)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_empty = fig.add_subplot(gs[0, 2])
    ax_empty.axis("off")
    ax_empty.text(
        0.0,
        0.72,
        "Overlay wiggle colors:\n"
        + "\n".join(f"{label}: {color}" for label, color in zip(labels, colors)),
        fontsize=11,
        va="top",
    )

    diff = gathers[1] - gathers[0]
    vmin_s, vmax_s = robust_clip(diff)
    receiver_max = (n_receivers - 1) * dx / 1000.0
    time_max = (nt - 1) * dt

    for shot_i in range(n_shots):
        for model_i in range(2):
            ax_img = fig.add_subplot(gs[shot_i + 1, model_i])
            im = ax_img.imshow(
                gathers[model_i][shot_i].T,
                cmap="seismic",
                origin="upper",
                aspect="auto",
                vmin=vmin_s,
                vmax=vmax_s,
                extent=[0, receiver_max, time_max, 0],
            )
            ax_img.set_title(f"{labels[model_i]} shot {shot_i}")
            ax_img.set_xlabel("Receiver x (km)")
            ax_img.set_ylabel("Time (s)")
            fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        ax_wig = fig.add_subplot(gs[shot_i + 1, 2])
        for gather, color, label in zip(gathers, colors, labels):
            plot_single_wiggle(ax_wig, gather[shot_i], color=color, dt=dt, dx=dx, scale=0.50)
        ax_wig.set_title(f"Shot {shot_i} wiggle overlay")
        ax_wig.set_xlim(-dx / 1000.0, receiver_max + dx / 1000.0)
        ax_wig.set_ylim(time_max, 0)
        ax_wig.set_xlabel("Receiver x (km)")
        ax_wig.set_ylabel("Time (s)")
        ax_wig.grid(True, color="0.9", linewidth=0.5)
        if shot_i == 0:
            ax_wig.legend(labels, loc="lower right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def infer_run_label(path: str, fallback: str) -> str:
    p = Path(path)
    config_path = p.parent / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("reward_tt_weight", 0) > 0:
                return "Travel-time inversion"
            fwi_type = str(config.get("fwi_type", "")).strip()
            if fwi_type:
                return f"{fwi_type} inversion"
        except Exception:
            pass
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B-spline smooth model and inversion wiggle comparison.")
    parser.add_argument("--cva_root", default="data/CVA/CurveVel_A")
    parser.add_argument("--file_idx", type=int, default=50)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--ctrl_h", type=int, default=4)
    parser.add_argument("--ctrl_w", type=int, default=4)
    parser.add_argument("--geometry", choices=["reflection", "transmission"], default="transmission")
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--n_receivers", type=int, default=70)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--freq", type=float, default=15.0)
    parser.add_argument("--pml_width", type=int, default=40)
    parser.add_argument("--simulator", choices=["deepwave", "fd"], default="deepwave")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compare_velocity", default="runs/phase4_gauss_logtt_s50/best_velocity.npy")
    parser.add_argument("--compare_label", default="")
    parser.add_argument("--out_single", default="figures/cva50_bspline_smooth_5shot_wiggle.png")
    parser.add_argument("--out_compare", default="figures/cva50_bspline_vs_tt_5shot_overlay.png")
    args = parser.parse_args()

    original, meta = load_cva_model(args.cva_root, args.file_idx, args.sample_idx)
    control, smooth = bspline_smooth_model(original, (args.ctrl_h, args.ctrl_w))
    compare = load_velocity(args.compare_velocity)
    compare_label = args.compare_label or infer_run_label(args.compare_velocity, "Inversion")

    print(f"Loaded CVA model: {meta}")
    print(f"Original range: {original.min():.2f}..{original.max():.2f}")
    print(f"B-spline control shape: {control.shape}, smooth range: {smooth.min():.2f}..{smooth.max():.2f}")
    print(f"Compare velocity: {args.compare_velocity}, range: {compare.min():.2f}..{compare.max():.2f}")
    print(f"Running {args.simulator} gathers on device={args.device}...")

    smooth_gathers = simulate_gathers(
        smooth,
        simulator=args.simulator,
        geometry=args.geometry,
        n_shots=args.n_shots,
        n_receivers=args.n_receivers,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
        freq=args.freq,
        pml_width=args.pml_width,
        device=args.device,
    )
    compare_gathers = simulate_gathers(
        compare,
        simulator=args.simulator,
        geometry=args.geometry,
        n_shots=args.n_shots,
        n_receivers=args.n_receivers,
        nt=args.nt,
        dx=args.dx,
        dt=args.dt,
        freq=args.freq,
        pml_width=args.pml_width,
        device=args.device,
    )

    plot_5shot_panel(
        smooth,
        smooth_gathers,
        title=f"CVA file_idx={args.file_idx}, sample={args.sample_idx}: 4x4 B-spline smooth model",
        out_path=Path(args.out_single),
        dt=args.dt,
        dx=args.dx,
        color="#2563eb",
    )
    plot_compare_panel(
        models=[smooth, compare],
        gathers=[smooth_gathers, compare_gathers],
        labels=["4x4 B-spline smooth target", compare_label],
        colors=["#2563eb", "#dc2626"],
        out_path=Path(args.out_compare),
        dt=args.dt,
        dx=args.dx,
    )

    print(f"Saved single panel: {Path(args.out_single).resolve()}")
    print(f"Saved compare panel: {Path(args.out_compare).resolve()}")


if __name__ == "__main__":
    main()
