from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def natural_key(path: Path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else path.stem


def ensure_shot_receiver_time(seismic: np.ndarray) -> np.ndarray:
    if seismic.ndim != 3:
        raise ValueError(f"seismic must be 3D, got shape={seismic.shape}")
    if seismic.shape[1] > seismic.shape[2]:
        seismic = np.transpose(seismic, (0, 2, 1))
    return np.ascontiguousarray(seismic)


def load_curvevel_a(root: str, file_idx: int, sample_idx: int) -> dict:
    root_path = Path(root)
    data_files = sorted((root_path / "data").glob("data*.npy"), key=natural_key)
    model_files = sorted((root_path / "model").glob("model*.npy"), key=natural_key)
    if not data_files or not model_files:
        raise FileNotFoundError(f"No data/model npy files found under {root_path}")
    if len(data_files) != len(model_files):
        raise ValueError("data/model file counts do not match")
    if file_idx < 0 or file_idx >= len(data_files):
        raise ValueError(f"file_idx={file_idx} outside [0, {len(data_files) - 1}]")

    data = np.load(data_files[file_idx], mmap_mode="r")
    model_data = np.load(model_files[file_idx], mmap_mode="r")
    if sample_idx < 0 or sample_idx >= data.shape[0]:
        raise ValueError(f"sample_idx={sample_idx} outside [0, {data.shape[0] - 1}]")

    seismic = np.asarray(data[sample_idx], dtype=np.float32)
    seismic = ensure_shot_receiver_time(seismic)
    peak = float(np.max(np.abs(seismic)))
    if peak > 0:
        seismic = seismic / peak

    model = np.asarray(model_data[sample_idx], dtype=np.float32)
    if model.ndim == 3 and model.shape[0] == 1:
        model = model[0]

    return {
        "seismic": seismic,
        "model": model,
        "meta": {
            "data_file": data_files[file_idx].name,
            "model_file": model_files[file_idx].name,
            "file_idx": int(file_idx),
            "sample_idx": int(sample_idx),
        },
    }


def robust_clip(data: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    limit = float(np.percentile(np.abs(data), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def plot_wiggle(
    ax: plt.Axes,
    shot: np.ndarray,
    dt: float,
    dx: float,
    scale: float,
    fill_positive: bool,
) -> None:
    """Plot a shot gather wiggle panel from [receiver, time] data."""
    n_receivers, nt = shot.shape
    time = np.arange(nt, dtype=np.float32) * float(dt)
    receiver_km = np.arange(n_receivers, dtype=np.float32) * float(dx) / 1000.0

    trace_peak = np.max(np.abs(shot), axis=1, keepdims=True)
    trace_peak = np.where(trace_peak > 1e-12, trace_peak, 1.0)
    normalized = shot / trace_peak
    spacing = float(dx) / 1000.0

    for i in range(n_receivers):
        x0 = receiver_km[i]
        trace = x0 + normalized[i] * spacing * float(scale)
        ax.plot(trace, time, color="black", linewidth=0.45)
        if fill_positive:
            ax.fill_betweenx(
                time,
                x0,
                trace,
                where=trace >= x0,
                color="black",
                alpha=0.35,
                linewidth=0,
            )

    ax.set_xlim(receiver_km[0] - spacing, receiver_km[-1] + spacing)
    ax.set_ylim(time[-1], time[0])
    ax.set_xlabel("Receiver x (km)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Wiggle plot")
    ax.grid(True, color="0.88", linewidth=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CurveVel_A shot gather wiggles.")
    parser.add_argument("--root", default="data/CVA/CurveVel_A")
    parser.add_argument("--file_idx", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--shot_idx", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument("--out", default="figures/cva_data1_sample0_shot0_wiggle.png")
    parser.add_argument("--no_fill", action="store_true")
    args = parser.parse_args()

    item = load_curvevel_a(args.root, args.file_idx, args.sample_idx)
    seismic = item["seismic"]
    model = item["model"]
    meta = item["meta"]

    if args.shot_idx < 0 or args.shot_idx >= seismic.shape[0]:
        raise ValueError(f"shot_idx={args.shot_idx} outside [0, {seismic.shape[0] - 1}]")

    shot = seismic[args.shot_idx]
    n_receivers, nt = shot.shape
    time_max = (nt - 1) * float(args.dt)
    receiver_max = (n_receivers - 1) * float(args.dx) / 1000.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    fig.suptitle(
        f"{meta['data_file']} sample {args.sample_idx}, shot {args.shot_idx}",
        fontsize=13,
    )

    im0 = axes[0].imshow(
        model,
        cmap="turbo",
        origin="upper",
        aspect="equal",
        extent=[0, model.shape[1] * args.dx / 1000.0, model.shape[0] * args.dx / 1000.0, 0],
    )
    axes[0].set_title("Velocity model")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("z (km)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("m/s")

    vmin, vmax = robust_clip(shot)
    im1 = axes[1].imshow(
        shot.T,
        cmap="seismic",
        origin="upper",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=[0, receiver_max, time_max, 0],
    )
    axes[1].set_title("Shot gather image")
    axes[1].set_xlabel("Receiver x (km)")
    axes[1].set_ylabel("Time (s)")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Normalized amplitude")

    plot_wiggle(
        axes[2],
        shot,
        dt=args.dt,
        dx=args.dx,
        scale=args.scale,
        fill_positive=not args.no_fill,
    )

    fig.savefig(out_path, dpi=220)
    print(f"Saved: {out_path.resolve()}")
    print(f"seismic shape [shot, receiver, time]: {seismic.shape}")
    print(f"model shape: {model.shape}")
    print(f"shot range: min={shot.min():.6g}, max={shot.max():.6g}")


if __name__ == "__main__":
    main()
