"""
Paper-quality visualization for VAE + latent RL seismic inversion.

Generates:
1. Velocity model comparison grid (True / VAE recon / Latent RL / B-spline RL / Error maps)
2. Vertical velocity profiles at key x-positions
3. MAE convergence curves (latent RL vs B-spline RL)
4. Reward convergence curves
5. Latent space evolution (optional)
"""
from __future__ import annotations
import sys, os, glob, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paper-ready style
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_run(run_dir: str) -> dict:
    """Load experiment data from a run directory."""
    result = {}
    # Load metrics
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        data = np.genfromtxt(metrics_path, delimiter=",", names=True)
        result["metrics"] = {name: data[name] for name in data.dtype.names}

    # Load config
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            result["config"] = json.load(f)

    # Load best velocity
    best_v_path = os.path.join(run_dir, "best_velocity.npy")
    if os.path.exists(best_v_path):
        result["best_velocity"] = np.load(best_v_path)

    return result


def plot_velocity_comparison(
    v_true: np.ndarray,
    v_vae_init: np.ndarray,
    v_latent_best: np.ndarray,
    v_bspline_best: np.ndarray,
    vae_mae: float,
    latent_mae: float,
    bspline_mae: float,
    save_path: str,
    model_name: str = "",
):
    """Figure 1: Velocity model comparison grid."""
    vmin = v_true.min()
    vmax = v_true.max()
    # Grid extent: 70 points × dx=10m = 700m = 0.7 km
    extent_km = [0, 0.7, 0.7, 0]  # x: 0→0.7km, z: 0→0.7km (surface at top)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Velocity models
    titles = [
        "True Velocity",
        f"Latent RL Best (MAE={latent_mae:.0f})",
        f"B-spline RL Best (MAE={bspline_mae:.0f})",
    ]
    models = [v_true, v_latent_best, v_bspline_best]
    for ax, title, model in zip(axes[0], titles, models):
        # model is [nz, nx]; imshow directly — z vertical, x horizontal
        im = ax.imshow(model, origin="upper", cmap="turbo", vmin=vmin, vmax=vmax,
                       aspect="equal", extent=extent_km)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")

    # Row 2: Error maps
    titles = ["", "|Latent − True|", "|B-spline − True|"]
    errors = [np.zeros_like(v_true), np.abs(v_latent_best - v_true), np.abs(v_bspline_best - v_true)]
    for ax, title, err in zip(axes[1], titles, errors):
        if title == "":
            # Show vertical profile comparison instead
            x_center = v_true.shape[0] // 2
            ax.plot(v_true[x_center, :], np.arange(v_true.shape[1]) * 0.01, "k-", label="True", linewidth=2)
            ax.plot(v_latent_best[x_center, :], np.arange(v_true.shape[1]) * 0.01, "b--", label="Latent RL", linewidth=1.5)
            ax.plot(v_bspline_best[x_center, :], np.arange(v_true.shape[1]) * 0.01, "r:", label="B-spline", linewidth=1.5)
            ax.invert_yaxis()
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Depth (km)")
            ax.set_title("Vertical Profile @ center")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            continue
        im = ax.imshow(err, origin="upper", cmap="hot",
                       aspect="equal", extent=extent_km)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Error (m/s)")

    fig.suptitle(f"Velocity Model Comparison — {model_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_convergence_curves(
    latent_metrics: dict,
    bspline_metrics: dict,
    save_path: str,
    model_name: str = "",
):
    """Figure 2: MAE and reward convergence curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE convergence
    ax = axes[0]
    if "best_mae_global" in latent_metrics:
        ax.plot(latent_metrics["step"], latent_metrics["best_mae_global"],
                "b-", label="Latent RL", linewidth=1.5)
    if bspline_metrics and "best_mae_global" in bspline_metrics:
        ax.plot(bspline_metrics["step"], bspline_metrics["best_mae_global"],
                "r--", label="B-spline RL", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Best MAE (m/s)")
    ax.set_title("MAE Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward L1
    ax = axes[1]
    if "reward_l1_mean" in latent_metrics:
        ax.plot(latent_metrics["step"], latent_metrics["reward_l1_mean"],
                "b-", label="Latent RL", linewidth=1.5)
    if bspline_metrics and "reward_l1_mean" in bspline_metrics:
        ax.plot(bspline_metrics["step"], bspline_metrics["reward_l1_mean"],
                "r--", label="B-spline RL", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("R_L1")
    ax.set_title("L1 Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratio + Entropy
    ax = axes[2]
    if "ratio_mean" in latent_metrics:
        ax.plot(latent_metrics["step"], latent_metrics["ratio_mean"],
                "b-", label="Latent ratio", linewidth=1)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("PPO Ratio Mean")
    ax.set_title("Policy Diagnostics")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — {model_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_summary_table(results: list[dict], save_path: str):
    """Figure 3: Summary bar chart across test models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [r["model"] for r in results]
    latent_maes = [r["latent_mae"] for r in results]
    bspline_maes = [r["bspline_mae"] for r in results]
    vae_mae = results[0].get("vae_val_mae", None)

    x = np.arange(len(models))
    width = 0.3

    bars1 = ax.bar(x - width/2, latent_maes, width, label="Latent RL (Ours)", color="#2196F3")
    bars2 = ax.bar(x + width/2, bspline_maes, width, label="B-spline RL", color="#FF5722")

    # VAE baseline as dashed line
    if vae_mae is not None:
        ax.axhline(y=vae_mae, color="green", linestyle="--", linewidth=1.5,
                   label=f"VAE reconstruction ({vae_mae:.0f})")
    # FWI baseline
    ax.axhline(y=141, color="purple", linestyle=":", linewidth=1.5,
               label="Gradient FWI (141)")

    for bar, mae in zip(bars1, latent_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{mae:.0f}", ha="center", fontsize=10, fontweight="bold")
    for bar, mae in zip(bars2, bspline_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{mae:.0f}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MAE (m/s)")
    ax.set_title("Multi-Model Generalization")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_runs", nargs="+", required=True,
                        help="Latent RL run dirs (e.g., runs/latent_200step_cva50)")
    parser.add_argument("--bspline_runs", nargs="+", required=True,
                        help="B-spline RL run dirs")
    parser.add_argument("--model_names", nargs="+", required=True,
                        help="Model labels")
    parser.add_argument("--cva_data", type=str, default="data/CVA/CurveVel_A",
                        help="CVA data root")
    parser.add_argument("--cva_indices", nargs="+", type=int, required=True,
                        help="CVA file indices")
    parser.add_argument("--vae_val_mae", type=float, default=113.5,
                        help="VAE validation MAE for reference line")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.data_loader import CurveVelADataset

    cva = CurveVelADataset(args.cva_data)
    summary_results = []

    for latent_dir, bspline_dir, name, fidx in zip(
        args.latent_runs, args.bspline_runs, args.model_names, args.cva_indices
    ):
        print(f"\n{'='*50}")
        print(f"Model: {name} (CVA[{fidx}])")

        # Load true velocity
        item = cva.get_by_file_index(fidx, 0)
        v_true = item["model"]

        # Load run data
        latent = load_run(latent_dir)
        bspline = load_run(bspline_dir)

        latent_mae = float(latent["metrics"]["best_mae_global"][-1]) if "best_mae_global" in latent.get("metrics", {}) else 999
        # Get best MAE across all steps
        if "best_mae_global" in latent.get("metrics", {}):
            latent_mae = float(latent["metrics"]["best_mae_global"].min())
        bspline_mae = 999
        if "best_mae_global" in bspline.get("metrics", {}):
            bspline_mae = float(bspline["metrics"]["best_mae_global"].min())

        print(f"  Latent RL best MAE: {latent_mae:.1f}")
        print(f"  B-spline RL best MAE: {bspline_mae:.1f}")

        v_latent = latent.get("best_velocity", v_true.copy())
        v_bspline = bspline.get("best_velocity", v_true.copy())

        # Figure 1: Velocity comparison
        plot_velocity_comparison(
            v_true=v_true,
            v_vae_init=v_true,  # placeholder — need actual VAE init
            v_latent_best=v_latent,
            v_bspline_best=v_bspline,
            vae_mae=args.vae_val_mae,
            latent_mae=latent_mae,
            bspline_mae=bspline_mae,
            save_path=os.path.join(args.out_dir, f"velocity_{name}.png"),
            model_name=name,
        )

        # Figure 2: Convergence
        plot_convergence_curves(
            latent_metrics=latent.get("metrics", {}),
            bspline_metrics=bspline.get("metrics", {}),
            save_path=os.path.join(args.out_dir, f"curves_{name}.png"),
            model_name=name,
        )

        summary_results.append({
            "model": name,
            "latent_mae": latent_mae,
            "bspline_mae": bspline_mae,
            "vae_val_mae": args.vae_val_mae,
        })

    # Figure 3: Summary
    plot_summary_table(
        summary_results,
        save_path=os.path.join(args.out_dir, "summary_comparison.png"),
    )

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<15} {'Latent RL':>12} {'B-spline':>12} {'Improvement':>12}")
    print(f"{'-'*60}")
    for r in summary_results:
        impr = (1 - r["latent_mae"] / r["bspline_mae"]) * 100 if r["bspline_mae"] > 0 else 0
        print(f"{r['model']:<15} {r['latent_mae']:>10.1f}  {r['bspline_mae']:>10.1f}  {impr:>9.1f}%")
    print(f"\nFigures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
