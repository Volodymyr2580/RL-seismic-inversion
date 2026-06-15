from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))


def robust_limit(x: np.ndarray, percentile: float = 99.0) -> float:
    value = float(np.percentile(np.abs(x), percentile))
    return value if np.isfinite(value) and value > 0 else 1.0


def source_positions(nx: int, n_shots: int) -> np.ndarray:
    if n_shots <= 1:
        return np.array([nx // 2], dtype=np.int64)
    return np.rint(np.linspace(0, nx - 1, n_shots)).astype(np.int64)


def receiver_positions(nx: int, n_receivers: int) -> np.ndarray:
    if n_receivers <= 1:
        return np.array([nx // 2], dtype=np.int64)
    return np.rint(np.linspace(0, nx - 1, n_receivers)).astype(np.int64)


def annotation_text(name: str, meta: dict, v_true: np.ndarray) -> str:
    nx, nz = v_true.shape
    dx = float(meta.get("dx", 10.0))
    dz = float(meta.get("dz", dx))
    n_shots = int(meta.get("n_shots", 0))
    n_receivers = int(meta.get("n_receivers", 0))
    source_depth = str(meta.get("source_depth", "bottom"))
    receiver_depth = str(meta.get("receiver_depth", "top"))
    nt = int(meta.get("nt", 0))
    dt = float(meta.get("dt", 0.0))
    freq = float(meta.get("freq", 0.0))

    lines = [
        name,
        f"model: nx={nx}, nz={nz}, dx={dx:g} m, dz={dz:g} m",
        f"survey: {n_shots} bottom sources -> {n_receivers} surface receivers",
        f"depths: source={source_depth}, receiver={receiver_depth}",
        f"time: nt={nt}, dt={dt:g} s, freq={freq:g} Hz",
    ]
    if "crop_shape_nx_nz" in meta:
        crop = meta["crop_shape_nx_nz"]
        lines.append(f"crop/source shape: nx={crop[0]}, nz={crop[1]}")
    if "dx_original" in meta and "dz_original" in meta:
        lines.append(f"source grid: dx={meta['dx_original']} m, dz={meta['dz_original']} m")
    if "ctrl_shape" in meta:
        ctrl = meta["ctrl_shape"]
        lines.append(f"B-spline ctrl: {ctrl[0]}x{ctrl[1]} ([nx,nz])")
    return "\n".join(lines)


def draw_acquisition(ax: plt.Axes, nx: int, nz: int, n_shots: int, n_receivers: int) -> None:
    src_x = source_positions(nx, n_shots)
    rec_x = receiver_positions(nx, n_receivers)
    rec_step = max(1, len(rec_x) // 90)
    ax.scatter(src_x, np.full_like(src_x, nz - 1), marker="v", s=58, c="#dc2626", edgecolors="white", linewidths=0.5, label="sources")
    ax.scatter(rec_x[::rec_step], np.zeros_like(rec_x[::rec_step]), marker="^", s=18, c="#06b6d4", edgecolors="black", linewidths=0.2, label="receivers")
    ax.plot([0, nx - 1], [0, 0], color="#06b6d4", linewidth=1.2, alpha=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=True)


def plot_model_panel(name: str, run_dir: Path, out_path: Path) -> None:
    meta = load_meta(run_dir)
    base = np.load(run_dir / "base_model.npy")
    v_true = np.load(run_dir / "v_true.npy")
    uniform = np.load(run_dir / "uniform_velocity.npy")
    nx, nz = v_true.shape

    all_v = np.stack([base, v_true, uniform])
    vmin, vmax = float(all_v.min()), float(all_v.max())
    residual = v_true - uniform
    emax = float(np.max(np.abs(residual)))

    fig = plt.figure(figsize=(17, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.82])
    fig.suptitle(f"Phase7 model and acquisition: {name}", fontsize=14)

    panels = [
        ("resampled input", base, "turbo", vmin, vmax),
        ("B-spline v_true + acquisition", v_true, "turbo", vmin, vmax),
        ("v_true - uniform", residual, "RdBu_r", -emax, emax),
    ]
    for i, (title, arr, cmap, lo, hi) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(arr.T, origin="upper", cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        if i == 1:
            draw_acquisition(ax, nx, nz, int(meta.get("n_shots", 0)), int(meta.get("n_receivers", 0)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m/s")

    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.axis("off")
    ax_info.text(
        0.02,
        0.98,
        annotation_text(name, meta, v_true),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round,pad=0.45"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sparse_global_wiggle(
    ax: plt.Axes,
    true_shot: np.ndarray,
    uniform_shot: np.ndarray,
    *,
    dt: float,
    dx: float,
    scale: float,
    max_traces: int = 55,
) -> None:
    n_receivers, nt = true_shot.shape
    time = np.arange(nt, dtype=np.float32) * float(dt)
    receiver_km = np.arange(n_receivers, dtype=np.float32) * float(dx) / 1000.0
    spacing = float(dx) / 1000.0
    step = max(1, n_receivers // max_traces)
    for i in range(0, n_receivers, step):
        x0 = receiver_km[i]
        ax.plot(np.full_like(time, x0), time, color="0.88", linewidth=0.25, alpha=0.65)
        ax.plot(x0 + true_shot[i] * spacing * scale, time, color="black", linewidth=0.55, alpha=0.85)
        ax.plot(x0 + uniform_shot[i] * spacing * scale, time, color="#dc2626", linewidth=0.55, alpha=0.75)
    ax.set_xlim(-spacing, receiver_km[-1] + spacing)
    ax.set_ylim(time[-1], 0)
    ax.set_xlabel("Receiver x (km)")
    ax.set_ylabel("Time (s)")
    ax.grid(True, color="0.9", linewidth=0.5)


def plot_gather_compare(name: str, run_dir: Path, out_path: Path) -> None:
    meta = load_meta(run_dir)
    true = np.load(run_dir / "gather_v_true_norm.npy")
    uniform = np.load(run_dir / "gather_uniform_norm.npy")
    diff = true - uniform
    dt = float(meta.get("dt", 0.001))
    dx = float(meta.get("dx", 10.0))
    n_shots, n_receivers, nt = true.shape

    shot_ids = [0, n_shots // 2, n_shots - 1] if n_shots >= 3 else list(range(n_shots))
    time_max = (nt - 1) * dt
    receiver_max = (n_receivers - 1) * dx / 1000.0
    gather_lim = robust_limit(np.concatenate([true.ravel(), uniform.ravel()]), 99.5)
    diff_lim = robust_limit(diff, 99.0)

    fig = plt.figure(figsize=(17, 3.55 * len(shot_ids)), constrained_layout=True)
    gs = fig.add_gridspec(len(shot_ids), 4, width_ratios=[1, 1, 1, 1.25])
    rel = meta.get("waveform_rel_l2", float("nan"))
    corr = meta.get("waveform_corr", float("nan"))
    fig.suptitle(f"Phase7 global-scale gather comparison: {name} (rel_l2={rel:.4g}, corr={corr:.4g})", fontsize=14)

    for row, shot_i in enumerate(shot_ids):
        panels = [
            ("v_true gather", true[shot_i], -gather_lim, gather_lim),
            ("uniform gather", uniform[shot_i], -gather_lim, gather_lim),
            ("v_true - uniform", diff[shot_i], -diff_lim, diff_lim),
        ]
        for col, (title, data, lo, hi) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                data.T,
                origin="upper",
                aspect="auto",
                cmap="seismic",
                vmin=lo,
                vmax=hi,
                extent=[0, receiver_max, time_max, 0],
            )
            ax.set_title(f"Shot {shot_i}: {title}")
            ax.set_xlabel("Receiver x (km)")
            ax.set_ylabel("Time (s)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax_wig = fig.add_subplot(gs[row, 3])
        plot_sparse_global_wiggle(ax_wig, true[shot_i], uniform[shot_i], dt=dt, dx=dx, scale=6.0)
        ax_wig.set_title(f"Shot {shot_i}: sparse global-scale wiggle")
        if row == 0:
            ax_wig.plot([], [], color="black", label="v_true")
            ax_wig.plot([], [], color="#dc2626", label="uniform")
            ax_wig.legend(loc="lower right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def default_runs(project_root: Path) -> dict[str, Path]:
    base = project_root / "outputs" / "phase7_sensitivity"
    return {
        "marmousi": base / "marmousi_nx210_nz70_ctrl30x10",
        "salt_A_90x270": base / "salt_nx270_nz90_ctrl36x12",
        "synthetic": base / "synthetic_nx210_nz70_ctrl30x10",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Make annotated Phase7 model and global gather comparison panels.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    out_base = Path(args.out_dir) if args.out_dir else project_root / "outputs" / "phase7_sensitivity" / "compare_panels"
    runs = default_runs(project_root)

    for name, run_dir in runs.items():
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        model_path = out_base / name / "model_annotated.png"
        gather_path = out_base / name / "global_gather_compare.png"
        plot_model_panel(name, run_dir, model_path)
        plot_gather_compare(name, run_dir, gather_path)
        print(model_path.resolve())
        print(gather_path.resolve())


if __name__ == "__main__":
    main()
