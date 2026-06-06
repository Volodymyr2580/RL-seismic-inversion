from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_velocity(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    v = np.load(path).astype(np.float32)
    if v.ndim == 3 and v.shape[0] == 1:
        v = v[0]
    if v.shape != (70, 70):
        raise ValueError(f"Expected velocity shape (70, 70), got {v.shape}: {path}")
    return v


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    out: dict[str, np.ndarray] = {}
    if not rows:
        return out
    for key in rows[0]:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=np.float64)
    return out


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def robust_abs_limit(data: np.ndarray, percentile: float = 99.0) -> float:
    limit = float(np.percentile(np.abs(data), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return limit


def simulate_gathers(
    velocity: np.ndarray,
    *,
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
    v_t = torch.from_numpy(velocity.astype(np.float32))
    gathers = forward.simulate(v_t, device=device).detach().cpu().numpy().astype(np.float32)
    if gathers.shape == (n_shots, nt, n_receivers):
        gathers = np.transpose(gathers, (0, 2, 1))
    elif gathers.shape != (n_shots, n_receivers, nt):
        raise ValueError(f"Expected [shot, receiver, time], got {gathers.shape}")
    peak = float(np.max(np.abs(gathers)))
    if peak > 0:
        gathers = gathers / peak
    return gathers


def plot_models_and_curves(
    run_dir: Path,
    models: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    true = models["true"]
    init = models["initial"]
    best = models["best_mae"]
    final = models["final"]
    all_v = np.stack([init, true, best, final])
    vmin, vmax = float(all_v.min()), float(all_v.max())

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.85])
    fig.suptitle(run_dir.name, fontsize=14)

    model_items = [
        ("Initial model", init),
        ("True model", true),
        ("Best MAE model", best),
        ("Final converged model", final),
    ]
    for col, (title, model) in enumerate(model_items):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(model, origin="upper", cmap="turbo", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    residual_items = [
        ("|Best MAE - True|", np.abs(best - true)),
        ("|Final - Best MAE|", np.abs(final - best)),
    ]
    for col, (title, residual) in enumerate(residual_items):
        ax = fig.add_subplot(gs[1, col * 2 : col * 2 + 2])
        im = ax.imshow(residual, origin="upper", cmap="magma", aspect="equal")
        ax.set_title(f"{title}, mean={residual.mean():.2f}")
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m/s")

    ax_reward = fig.add_subplot(gs[2, :2])
    if "step" in metrics and "reward_l1_mean" in metrics and np.nanmax(np.abs(metrics["reward_l1_mean"])) > 0:
        ax_reward.plot(metrics["step"], metrics["reward_l1_mean"], color="#16a34a", label="L1")
    if "step" in metrics and "reward_l2_mean" in metrics and np.nanmax(np.abs(metrics["reward_l2_mean"])) > 0:
        ax_reward.plot(metrics["step"], metrics["reward_l2_mean"], color="#111827", label="FWI")
    if "reward_tt_mean" in metrics and np.nanmax(np.abs(metrics["reward_tt_mean"])) > 0:
        ax_reward.plot(metrics["step"], metrics["reward_tt_mean"], color="#2563eb", label="TT")
    ax_reward.set_title("Reward curve")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend()

    ax_mae = fig.add_subplot(gs[2, 2:])
    if "step" in metrics:
        if "mae_oracle_best" in metrics:
            ax_mae.plot(metrics["step"], metrics["mae_oracle_best"], color="#dc2626", label="oracle best in group")
        if "best_mae_global" in metrics:
            ax_mae.plot(metrics["step"], metrics["best_mae_global"], color="#111827", label="global best")
    ax_mae.set_title("MAE convergence")
    ax_mae.set_xlabel("Step")
    ax_mae.set_ylabel("MAE (m/s)")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_wiggle_trace(ax: plt.Axes, shot: np.ndarray, *, color: str, dt: float, dx: float, scale: float) -> None:
    n_receivers, nt = shot.shape
    time = np.arange(nt, dtype=np.float32) * float(dt)
    receiver_km = np.arange(n_receivers, dtype=np.float32) * float(dx) / 1000.0
    spacing = float(dx) / 1000.0
    peak = np.max(np.abs(shot), axis=1, keepdims=True)
    normalized = shot / np.where(peak > 1e-12, peak, 1.0)
    for i in range(n_receivers):
        x0 = receiver_km[i]
        ax.plot(x0 + normalized[i] * spacing * scale, time, color=color, linewidth=0.42, alpha=0.78)


def plot_wiggle_overlays(
    run_dir: Path,
    gathers: dict[str, np.ndarray],
    out_path: Path,
    *,
    dt: float,
    dx: float,
) -> None:
    true = gathers["true"]
    candidates = [
        ("Initial vs true", gathers["initial"]),
        ("Best MAE vs true", gathers["best_mae"]),
        ("Final vs true", gathers["final"]),
    ]
    n_shots, n_receivers, nt = true.shape
    receiver_max = (n_receivers - 1) * dx / 1000.0
    time_max = (nt - 1) * dt

    fig = plt.figure(figsize=(17, 3.0 * n_shots), constrained_layout=True)
    gs = fig.add_gridspec(n_shots, len(candidates))
    fig.suptitle(f"{run_dir.name}: shot gather wiggle overlays", fontsize=14)

    for shot_i in range(n_shots):
        for col, (title, candidate) in enumerate(candidates):
            ax = fig.add_subplot(gs[shot_i, col])
            plot_wiggle_trace(ax, true[shot_i], color="black", dt=dt, dx=dx, scale=0.46)
            plot_wiggle_trace(ax, candidate[shot_i], color="#dc2626", dt=dt, dx=dx, scale=0.46)
            ax.set_title(f"Shot {shot_i}: {title}")
            ax.set_xlim(-dx / 1000.0, receiver_max + dx / 1000.0)
            ax.set_ylim(time_max, 0)
            ax.set_xlabel("Receiver x (km)")
            ax.set_ylabel("Time (s)")
            ax.grid(True, color="0.9", linewidth=0.5)
            if shot_i == 0:
                ax.plot([], [], color="black", label="true")
                ax.plot([], [], color="#dc2626", label="candidate")
                ax.legend(loc="lower right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize_run(run_dir: Path, device: str) -> None:
    config = load_config(run_dir / "config.json")
    models = {
        "initial": load_velocity(run_dir / "init_velocity.npy"),
        "true": load_velocity(run_dir / "true_velocity.npy"),
        "best_mae": load_velocity(run_dir / "best_velocity.npy"),
        "final": load_velocity(run_dir / "final_velocity.npy"),
    }
    metrics = read_metrics(run_dir / "metrics.csv")

    n_shots = int(config.get("n_shots", 5))
    n_receivers = int(config.get("n_receivers", 70))
    nt = int(config.get("nt", 1000))
    dx = float(config.get("dx", 10.0))
    dt = float(config.get("dt", 0.001))
    freq = float(config.get("freq", 15.0))
    pml_width = int(config.get("pml_width", 40))
    geometry = str(config.get("geometry", "transmission"))

    viz_dir = run_dir / "phase6_visuals"
    plot_models_and_curves(run_dir, models, metrics, viz_dir / "models_residuals_curves.png")

    gathers = {
        name: simulate_gathers(
            model,
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
        for name, model in models.items()
    }
    plot_wiggle_overlays(run_dir, gathers, viz_dir / "shot_wiggle_overlays.png", dt=dt, dx=dx)
    print(f"Saved Phase6 visuals under {viz_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Phase6 single-reward RL convergence runs.")
    parser.add_argument("--run_dir", required=True, help="A single Phase6 run directory.")
    parser.add_argument("--device", default="auto", help="deepwave device: auto, cpu, or cuda.")
    args = parser.parse_args()
    visualize_run(Path(args.run_dir), device=args.device)


if __name__ == "__main__":
    main()
