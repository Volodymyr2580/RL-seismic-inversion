"""Prepare deterministic Phase7 initial models.

The training suite uses three non-random starts for each true model:
far_uniform, medium_gradient, and near_blur. Velocity arrays are stored as
[nx, nz], matching the rest of this project. Plots transpose to [nz, nx].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    {
        "name": "marmousi",
        "v_true": "outputs/phase7_sensitivity/marmousi_nx210_nz70_ctrl30x10/v_true.npy",
        "nx": 210,
        "nz": 70,
        "dx": 10.0,
        "dz": 10.0,
        "ctrl": "30x10",
        "acquisition": "bottom sources, surface receivers, n_shots=7, n_receivers=210",
    },
    {
        "name": "salt_A_90x270",
        "v_true": "outputs/phase7_sensitivity/salt_nx270_nz90_ctrl36x12/v_true.npy",
        "nx": 270,
        "nz": 90,
        "dx": 10.0,
        "dz": 10.0,
        "ctrl": "36x12",
        "acquisition": "bottom sources, surface receivers, n_shots=7, n_receivers=270",
    },
    {
        "name": "synthetic",
        "v_true": "outputs/phase7_sensitivity/synthetic_nx210_nz70_ctrl30x10/v_true.npy",
        "nx": 210,
        "nz": 70,
        "dx": 10.0,
        "dz": 10.0,
        "ctrl": "30x10",
        "acquisition": "bottom sources, surface receivers, n_shots=7, n_receivers=210",
    },
]


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(max(sigma, 1e-6))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def _convolve_axis_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    arr_pad = np.pad(arr, [(pad, pad) if i == axis else (0, 0) for i in range(arr.ndim)], mode="reflect")
    moved = np.moveaxis(arr_pad, axis, 0)
    out = np.empty_like(np.moveaxis(arr, axis, 0), dtype=np.float32)
    for i in range(out.shape[0]):
        window = moved[i : i + len(kernel)]
        out[i] = np.tensordot(kernel, window, axes=(0, 0))
    return np.moveaxis(out, 0, axis)


def gaussian_blur(arr: np.ndarray, sigma_x: float, sigma_z: float) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    out = _convolve_axis_reflect(out, _gaussian_kernel1d(sigma_x), axis=0)
    out = _convolve_axis_reflect(out, _gaussian_kernel1d(sigma_z), axis=1)
    return out.astype(np.float32)


def make_initial_models(v_true: np.ndarray) -> dict[str, np.ndarray]:
    nx, nz = v_true.shape
    vmin, vmax = float(v_true.min()), float(v_true.max())

    far_uniform = np.full_like(v_true, float(v_true.mean()), dtype=np.float32)

    top = float(np.mean(v_true[:, : max(2, nz // 10)]))
    bottom = float(np.mean(v_true[:, -max(2, nz // 10) :]))
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    gradient_1d = top + (bottom - top) * z
    medium_gradient = np.repeat(gradient_1d[None, :], nx, axis=0).astype(np.float32)

    # Strong blur keeps large-scale structure but removes most high-frequency detail.
    near_blur = gaussian_blur(v_true, sigma_x=max(4.0, nx / 22.0), sigma_z=max(2.0, nz / 18.0))
    near_blur = np.clip(near_blur, vmin, vmax).astype(np.float32)

    return {
        "far_uniform": far_uniform,
        "medium_gradient": medium_gradient,
        "near_blur": near_blur,
    }


def save_preview(out_path: Path, model_name: str, v_true: np.ndarray, initials: dict[str, np.ndarray], meta: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = float(v_true.min()), float(v_true.max())
    nx, nz = v_true.shape
    extent = [0, nx * meta["dx"] / 1000.0, nz * meta["dz"] / 1000.0, 0]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

    panels = [("true", v_true), *initials.items()]
    for ax, (name, arr) in zip(axes[0], panels):
        mae = float(np.mean(np.abs(arr - v_true)))
        im = ax.imshow(arr.T, origin="upper", aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f"{name}\nMAE={mae:.1f} m/s" if name != "true" else "true")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        plt.colorbar(im, ax=ax, fraction=0.045, label="m/s")

    for ax, (name, arr) in zip(axes[1], panels):
        if name == "true":
            ax.axis("off")
            ax.text(
                0.02,
                0.82,
                f"{model_name}\n(nx,nz)=({nx},{nz})\n(dx,dz)=({meta['dx']},{meta['dz']}) m\nctrl={meta['ctrl']}\n{meta['acquisition']}",
                va="top",
                ha="left",
                fontsize=11,
                transform=ax.transAxes,
            )
            continue
        err = arr - v_true
        lim = float(np.percentile(np.abs(err), 99))
        im = ax.imshow(err.T, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=extent)
        ax.set_title(f"{name} - true")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        plt.colorbar(im, ax=ax, fraction=0.045, label="m/s")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Phase7 initial velocity models")
    parser.add_argument("--out_root", default=str(ROOT / "outputs" / "phase7_sensitivity" / "initial_models"))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    suite_meta = {"models": []}
    for model in DEFAULT_MODELS:
        v_path = ROOT / model["v_true"]
        if not v_path.exists():
            raise FileNotFoundError(f"Missing v_true for {model['name']}: {v_path}")
        v_true = np.load(v_path).astype(np.float32)
        expected = (int(model["nx"]), int(model["nz"]))
        if v_true.shape != expected:
            raise ValueError(f"{model['name']} shape {v_true.shape} != {expected}: {v_path}")

        out_dir = out_root / model["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        initials = make_initial_models(v_true)

        init_meta = {
            **model,
            "v_true_abs": str(v_path),
            "initial_models": {},
        }
        for init_name, arr in initials.items():
            save_path = out_dir / f"{init_name}.npy"
            np.save(save_path, arr.astype(np.float32))
            init_meta["initial_models"][init_name] = {
                "path": str(save_path),
                "mae_to_true": float(np.mean(np.abs(arr - v_true))),
                "v_min": float(arr.min()),
                "v_max": float(arr.max()),
                "description": {
                    "far_uniform": "constant velocity equal to the true-model spatial mean",
                    "medium_gradient": "laterally uniform vertical gradient estimated from top/bottom true-model averages",
                    "near_blur": "strong Gaussian-smoothed version of the true model",
                }[init_name],
            }

        preview_path = out_dir / "initial_models_preview.png"
        save_preview(preview_path, model["name"], v_true, initials, init_meta)
        init_meta["preview"] = str(preview_path)
        with (out_dir / "init_meta.json").open("w", encoding="utf-8") as f:
            json.dump(init_meta, f, indent=2, ensure_ascii=False)
        suite_meta["models"].append(init_meta)
        print(f"[ok] {model['name']}: {out_dir}")

    with (out_root / "phase7_initial_models.json").open("w", encoding="utf-8") as f:
        json.dump(suite_meta, f, indent=2, ensure_ascii=False)
    print(f"[ok] metadata: {out_root / 'phase7_initial_models.json'}")


if __name__ == "__main__":
    main()
