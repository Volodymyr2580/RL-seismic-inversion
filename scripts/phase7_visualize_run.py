"""Create Phase7 per-run visual diagnostics.

Usage:
    python scripts/phase7_visualize_run.py runs/phase7/marmousi/far_uniform/l1l2 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.transmission_forward import AcquisitionForward, AcquisitionGeometry


def load_config(run_dir: Path) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(cfg: dict, key: str) -> int:
    return int(cfg[key])


def _as_float(cfg: dict, key: str) -> float:
    return float(cfg[key])


def load_velocity(path: Path | str | None) -> np.ndarray | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def get_models(run_dir: Path, cfg: dict) -> dict[str, np.ndarray]:
    models: dict[str, np.ndarray] = {}
    true_v = load_velocity(run_dir / "true_velocity.npy")
    if true_v is not None:
        models["true"] = true_v

    init_path = cfg.get("init_velocity_path") or ""
    init_v = load_velocity(init_path)
    if init_v is None:
        init_v = load_velocity(run_dir / "init_velocity.npy")
    if init_v is not None:
        models["initial"] = init_v

    best_v = load_velocity(run_dir / "best_velocity.npy")
    if best_v is not None:
        models["best"] = best_v

    final_v = load_velocity(run_dir / "final_velocity.npy")
    if final_v is not None:
        models["final"] = final_v
    return models


def make_geom(cfg: dict) -> AcquisitionGeometry:
    return AcquisitionGeometry(
        nx_model=_as_int(cfg, "nx_model"),
        nz_model=_as_int(cfg, "nz_model"),
        dx=_as_float(cfg, "dx"),
        dt=_as_float(cfg, "dt"),
        freq=_as_float(cfg, "freq"),
        nt=_as_int(cfg, "nt"),
        n_shots=_as_int(cfg, "n_shots"),
        n_receivers=_as_int(cfg, "n_receivers"),
        pml_width=_as_int(cfg, "pml_width"),
        geometry=str(cfg.get("geometry", "transmission")),
        source_depth=str(cfg.get("source_depth", "auto")),
        receiver_depth=str(cfg.get("receiver_depth", "auto")),
    )


@torch.no_grad()
def simulate_gathers(models: dict[str, np.ndarray], cfg: dict, device_name: str) -> dict[str, np.ndarray]:
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    geom = make_geom(cfg)
    forward = AcquisitionForward(geom)
    gathers: dict[str, np.ndarray] = {}
    for name, arr in models.items():
        v = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        gathers[name] = forward.simulate(v, device=str(device)).detach().cpu().numpy()
    return gathers


def plot_models(out_dir: Path, run_dir: Path, cfg: dict, models: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "true" not in models:
        return
    v_true = models["true"]
    nx, nz = v_true.shape
    dx = _as_float(cfg, "dx")
    dz = dx
    extent = [0, nx * dx / 1000.0, nz * dz / 1000.0, 0]
    vmin, vmax = float(v_true.min()), float(v_true.max())
    names = [n for n in ["true", "initial", "best", "final"] if n in models]

    fig, axes = plt.subplots(2, len(names), figsize=(4.7 * len(names), 8.2), constrained_layout=True)
    if len(names) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, name in enumerate(names):
        arr = models[name]
        mae = float(np.mean(np.abs(arr - v_true)))
        im = axes[0, j].imshow(arr.T, origin="upper", aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax, extent=extent)
        axes[0, j].set_title(f"{name}\nMAE={mae:.1f} m/s" if name != "true" else "true")
        axes[0, j].set_xlabel("x (km)")
        axes[0, j].set_ylabel("z (km)")
        plt.colorbar(im, ax=axes[0, j], fraction=0.045, label="m/s")

        if name == "true":
            axes[1, j].axis("off")
            title = f"{run_dir.parts[-3:] if len(run_dir.parts) >= 3 else run_dir.name}"
            axes[1, j].text(
                0.02,
                0.92,
                f"{title}\n(nx,nz)=({nx},{nz})\n(dx,dz)=({dx},{dz}) m\n"
                f"shots={cfg.get('n_shots')}, receivers={cfg.get('n_receivers')}\n"
                f"source={cfg.get('source_depth')}, receiver={cfg.get('receiver_depth')}",
                va="top",
                ha="left",
                transform=axes[1, j].transAxes,
                fontsize=10,
            )
        else:
            err = arr - v_true
            lim = float(np.percentile(np.abs(err), 99))
            im = axes[1, j].imshow(err.T, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=extent)
            axes[1, j].set_title(f"{name} - true")
            axes[1, j].set_xlabel("x (km)")
            axes[1, j].set_ylabel("z (km)")
            plt.colorbar(im, ax=axes[1, j], fraction=0.045, label="m/s")
    fig.savefig(out_dir / "models.png", dpi=160)
    plt.close(fig)


def _global_clip(gathers: dict[str, np.ndarray], shot_idx: int) -> tuple[float, float]:
    vals = []
    for arr in gathers.values():
        vals.append(arr[shot_idx])
    cat = np.concatenate([v.reshape(-1) for v in vals])
    q = float(np.percentile(np.abs(cat), 99.3))
    return -q, q


def plot_gathers(out_dir: Path, cfg: dict, gathers: dict[str, np.ndarray], shot_idx: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ["true", "initial", "best", "final"] if n in gathers]
    if not names:
        return
    dx = _as_float(cfg, "dx")
    dt = _as_float(cfg, "dt")
    nrec = _as_int(cfg, "n_receivers")
    nt = _as_int(cfg, "nt")
    vmin, vmax = _global_clip(gathers, shot_idx)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 5.2), constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        g = gathers[name][shot_idx]
        im = ax.imshow(g.T, origin="upper", aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax, extent=[0, nrec * dx / 1000.0, nt * dt, 0])
        ax.set_title(f"{name} shot {shot_idx}")
        ax.set_xlabel("receiver x (km)")
        ax.set_ylabel("time (s)")
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.savefig(out_dir / "gather_images.png", dpi=160)
    plt.close(fig)

    if "true" in gathers:
        comp_names = [n for n in ["initial", "best", "final"] if n in gathers]
        fig, axes = plt.subplots(1, len(comp_names), figsize=(4.8 * max(1, len(comp_names)), 5.2), constrained_layout=True)
        if len(comp_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, comp_names):
            diff = gathers[name][shot_idx] - gathers["true"][shot_idx]
            q = float(np.percentile(np.abs(diff), 99.3))
            im = ax.imshow(diff.T, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-q, vmax=q, extent=[0, nrec * dx / 1000.0, nt * dt, 0])
            ax.set_title(f"{name} - true")
            ax.set_xlabel("receiver x (km)")
            ax.set_ylabel("time (s)")
            plt.colorbar(im, ax=ax, fraction=0.045)
        fig.savefig(out_dir / "gather_residuals.png", dpi=160)
        plt.close(fig)


def plot_wiggles(out_dir: Path, cfg: dict, gathers: dict[str, np.ndarray], shot_idx: int, max_traces: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "true" not in gathers:
        return
    comp_names = [n for n in ["initial", "best", "final"] if n in gathers]
    if not comp_names:
        return
    true_g = gathers["true"][shot_idx]
    nrec, nt = true_g.shape
    stride = max(1, int(np.ceil(nrec / max_traces)))
    rec_ids = np.arange(0, nrec, stride)
    t = np.arange(nt) * _as_float(cfg, "dt")
    x_offsets = rec_ids.astype(np.float32)
    global_amp = max(float(np.percentile(np.abs(true_g), 99.5)), 1e-12)
    for name in comp_names:
        global_amp = max(global_amp, float(np.percentile(np.abs(gathers[name][shot_idx]), 99.5)))
    scale = 0.42 * stride / global_amp

    fig, axes = plt.subplots(1, len(comp_names), figsize=(5.6 * len(comp_names), 6.4), constrained_layout=True)
    if len(comp_names) == 1:
        axes = [axes]
    colors = {"initial": "#1f77b4", "best": "#d62728", "final": "#2ca02c"}
    for ax, name in zip(axes, comp_names):
        pred_g = gathers[name][shot_idx]
        for offset, rid in zip(x_offsets, rec_ids):
            tr_true = true_g[rid] * scale + offset
            tr_pred = pred_g[rid] * scale + offset
            ax.plot(tr_true, t, color="black", linewidth=0.65, alpha=0.78)
            ax.plot(tr_pred, t, color=colors.get(name, "#d62728"), linewidth=0.65, alpha=0.78)
        ax.invert_yaxis()
        ax.set_title(f"Wiggle: true vs {name}")
        ax.set_xlabel("receiver index + scaled amplitude")
        ax.set_ylabel("time (s)")
        ax.grid(True, alpha=0.2)
    fig.savefig(out_dir / "wiggle_overlay.png", dpi=170)
    plt.close(fig)


def read_metrics(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if v != ""})
    return rows


def plot_metrics(out_dir: Path, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_metrics(run_dir)
    if not rows:
        return
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [r.get("best_mae_global", np.nan) for r in rows], label="best_mae_global")
    axes[0, 0].plot(steps, [r.get("mae_oracle_best", np.nan) for r in rows], label="mae_oracle_best", alpha=0.65)
    axes[0, 0].set_title("MAE")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, [r.get("reward_l2_mean", np.nan) for r in rows], label="reward_l2")
    axes[0, 1].plot(steps, [r.get("reward_l1_mean", np.nan) for r in rows], label="reward_l1")
    axes[0, 1].plot(steps, [r.get("reward_tt_mean", np.nan) for r in rows], label="reward_tt")
    axes[0, 1].set_title("Rewards")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, [r.get("entropy", np.nan) for r in rows])
    axes[1, 0].set_title("Entropy")
    axes[1, 0].set_xlabel("step")

    axes[1, 1].plot(steps, [r.get("ratio_mean", np.nan) for r in rows], label="ratio_mean")
    axes[1, 1].plot(steps, [r.get("clip_frac", np.nan) for r in rows], label="clip_frac")
    axes[1, 1].axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axes[1, 1].set_title("PPO Ratio Diagnostics")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()
    fig.savefig(out_dir / "metrics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one Phase7 reward-suite run")
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shot_idx", type=int, default=-1, help="-1 means middle shot")
    parser.add_argument("--max_traces", type=int, default=35)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json: {run_dir}")
    cfg = load_config(run_dir)
    out_dir = run_dir / "phase7_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = get_models(run_dir, cfg)
    if "true" not in models:
        raise FileNotFoundError(f"Missing true_velocity.npy in {run_dir}")
    shot_idx = args.shot_idx
    if shot_idx < 0:
        shot_idx = _as_int(cfg, "n_shots") // 2

    plot_models(out_dir, run_dir, cfg, models)
    plot_metrics(out_dir, run_dir)
    gathers = simulate_gathers(models, cfg, args.device)
    plot_gathers(out_dir, cfg, gathers, shot_idx)
    plot_wiggles(out_dir, cfg, gathers, shot_idx, args.max_traces)

    manifest = {
        "run_dir": str(run_dir),
        "visual_dir": str(out_dir),
        "shot_idx": shot_idx,
        "files": sorted(p.name for p in out_dir.glob("*.png")),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[ok] visuals: {out_dir}")


if __name__ == "__main__":
    main()
