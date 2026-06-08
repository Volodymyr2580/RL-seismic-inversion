"""
Phase II training entry: Multi-objective RL-FWI with Beta policy + GDPO-Guard.

Transmission geometry: sources at surface, receivers at model bottom (not in PML).
Reward: L1 and L2 data misfit as separate GDPO components.

Usage:
    # Quick smoke test (synthetic layered model, learnable policy)
    python train_rl_fwi.py --steps 50 --group_size 8 --device cpu

    # CNN-conditioned policy, Marmousi from CVA
    python train_rl_fwi.py --model_source cva --cva_file_idx 0 --cva_sample_idx 0 \\
        --policy_type cnn --steps 2000 --group_size 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Suppress deepwave grid-per-wavelength warnings
import warnings
warnings.filterwarnings("ignore", message=".*grid cells per wavelength.*")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.beta_policy import (
    BetaSplinePolicy,
    CNNBetaMeanPolicy,
    GaussianMeanPolicy,
    LearnableBetaSplinePolicy,
    LearnableBetaMeanPolicy,
    unit_to_velocity,
)
from agents.velocity_reconstructor import VelocityReconstructor
from agents.transmission_forward import AcquisitionForward, AcquisitionGeometry, TransmissionForward, TransmissionGeometry
from agents.seismic_layout import assert_batch_shot_receiver_time, assert_shot_receiver_time
from agents.rl_objectives import (
    RewardWeights,
    gdpo_advantage,
    RatioGuardState,
    clipped_policy_loss,
    reward_l1,
    reward_l2,
    sign_preserving_log,
)


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model geometry
    nx_model: int = 70
    nz_model: int = 70
    nx_ctrl: int = 4
    nz_ctrl: int = 4
    v_min: float = 1500.0
    v_max: float = 4500.0

    # Acquisition (transmission or reflection)
    geometry: str = "reflection"
    n_shots: int = 5
    n_receivers: int = 70
    nt: int = 1000
    dx: float = 10.0
    dt: float = 0.001
    freq: float = 15.0
    pml_width: int = 40

    # Policy
    policy_type: str = "mean"  # "mean" (μ+κ), "learnable" (α+β legacy), or "cnn"
    cnn_embed_dim: int = 128
    init_alpha: float = 2.0
    init_beta: float = 2.0
    init_kappa: float = 4.0  # for mean policy: α+β concentration
    pretrain_ckpt: Optional[str] = None
    pretrain_mu_clip: float = 0.05
    resume_ckpt: Optional[str] = None  # resume policy from checkpoint (for progressive)
    vae_ckpt: Optional[str] = None
    latent_dim: int = 64
    unfreeze_decoder: int = 0  # unfreeze last N decoder layers

    # RL
    group_size: int = 8
    steps: int = 500
    ppo_epochs: int = 4
    lr: float = 5e-3
    optimizer: str = "adamw"  # adamw or muon
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0

    # Clipping
    epsilon_low: float = 0.20
    epsilon_high: float = 0.27

    # Reward weights
    reward_l1_weight: float = 1.0
    reward_l2_weight: float = 1.0
    fwi_type: str = "l2"  # l2, envelope, windowed_l2, wasserstein, contrastive, ncc_*, awi, phase_func
    fwi_type2: str = ""    # optional second FWI reward for multi-FWI mixing (empty = disabled)
    fwi_weight2: float = 0.0  # weight for second FWI reward in GDPO
    wasserstein_normalize: str = "abs"
    ncc_lag_max: int = 80              # max lag for ncc_maxlag (samples)
    ncc_lag_penalty: float = 0.005      # penalty weight for nonzero lag
    awi_version: str = "l1"            # "l1" | "full"
    freq_band: str = ""   # e.g. "0-5" or "0-10" — lowpass cutoff in Hz before reward
    reward_si_weight: float = 0.0
    reward_prior_weight: float = 0.05
    reward_tt_weight: float = 0.0  # travel-time reward weight
    reward_tt_log: bool = False    # use log-scaled tt reward (amplifies small |Δt|)
    si_every: int = 10  # compute SI every N steps (expensive)

    # Best criterion
    best_criterion: str = "l2"  # "mae", "l2", or "si"
    init_temperature: float = 2.0
    final_temperature: float = 0.1
    anneal_steps: int = 400

    # Entropy bonus (prevent Beta mode collapse)
    entropy_bonus: float = 0.02

    # GRPO-Guard (ratio monitoring, not used in REINFORCE phase)
    guard_enabled: bool = True
    guard_threshold: float = 0.05

    # Model source
    model_source: str = "synthetic"  # "synthetic", "cva", "fva", or "smooth"
    cva_root: str = "data/CVA/CurveVel_A"
    cva_file_idx: int = 0
    cva_sample_idx: int = 0
    fva_root: str = "data/FVA_model"
    smooth_root: str = "data/smooth_models"

    # Early stopping
    early_stop_patience: int = 500  # stop if no MAE improvement for N steps
    early_stop_window: int = 50     # smoothing window for reward

    # Output
    out_dir: str = "runs/phase2"
    save_every: int = 50
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
#  Synthetic velocity model
# ---------------------------------------------------------------------------

def make_synthetic_layered_model(
    nx: int = 70,
    nz: int = 70,
    v_min: float = 1500.0,
    v_max: float = 4500.0,
) -> np.ndarray:
    """Create a simple 3-layer velocity model with horizontal interfaces.

    Layer 1 (top):    v_min ~ 1500-2000
    Layer 2 (middle): 3000-3500
    Layer 3 (bottom): v_max ~ 4000-4500
    """
    v = np.zeros((nx, nz), dtype=np.float32)
    layer1_bottom = int(nz * 0.2)
    layer2_bottom = int(nz * 0.6)

    v1 = v_min + 500
    v2 = 3200.0
    v3 = v_max - 200

    v[:, :layer1_bottom] = v1
    v[:, layer1_bottom:layer2_bottom] = v2
    v[:, layer2_bottom:] = v3

    # Smooth interfaces slightly
    from scipy.ndimage import gaussian_filter
    v = gaussian_filter(v, sigma=1.0)
    return v.astype(np.float32)


def make_observation_from_velocity(
    v_true: np.ndarray,
    geom: AcquisitionGeometry,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic observation data from a true velocity model.

    Returns:
        p_data: [n_shots, n_receivers, nt] observed seismogram.
        v_true_t: [nx, nz] true velocity as tensor.
    """
    forward = AcquisitionForward(geom)
    v_true_t = torch.from_numpy(v_true).to(device=device, dtype=torch.float32)
    p_data = forward.simulate(v_true_t, device=device)
    return p_data, v_true_t


# ---------------------------------------------------------------------------
#  Compute rewards for a group of predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_group_rewards(
    p_pred: torch.Tensor,
    p_obs: torch.Tensor,
    weights: RewardWeights,
) -> dict[str, torch.Tensor]:
    """Compute L1 and L2 rewards for G predictions against a single observation.

    Args:
        p_pred: [G, n_shots, n_receivers, nt] predicted seismograms.
        p_obs:  [n_shots, n_receivers, nt] observed seismogram.
        weights: reward weights config.

    Returns:
        dict with 'l1' and 'l2' keys, each [G] tensor.
    """
    g = int(p_pred.shape[0])
    # Broadcast observation to match group
    p_obs_batch = p_obs.unsqueeze(0).expand(g, -1, -1, -1)

    rewards = {}
    if weights.l1 != 0.0:
        rewards["l1"] = reward_l1(p_pred, p_obs_batch)  # [G]
    if weights.l2 != 0.0:
        rewards["l2"] = reward_l2(p_pred, p_obs_batch)  # [G]

    return rewards


def init_mean_policy_from_cnn_pretrain(
    policy: LearnableBetaMeanPolicy,
    ckpt_path: str,
    p_data: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> None:
    """Initialize 32-parameter mean policy from a pretrained CNN predictor.

    The pretrained CNN maps the current observation to 4×4 control-point means.
    We copy those means into the unconditional LearnableBetaMeanPolicy and then
    let the compact 32-parameter policy handle RL fine-tuning.
    """
    init_policy = CNNBetaMeanPolicy(
        nx_ctrl=config.nx_ctrl,
        nz_ctrl=config.nz_ctrl,
        in_channels=config.n_shots,
        embed_dim=config.cnn_embed_dim,
        v_min=config.v_min,
        v_max=config.v_max,
        init_kappa=config.init_kappa,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    init_policy.load_pretrained_velocity_predictor(state_dict)
    init_policy.eval()

    with torch.no_grad():
        alpha, beta_param = init_policy(p_data.unsqueeze(0).to(device))
        mu_clip = min(max(float(config.pretrain_mu_clip), 1e-4), 0.49)
        u_mean = (alpha / (alpha + beta_param)).squeeze(0).clamp(mu_clip, 1.0 - mu_clip)
        policy.mu_raw.copy_(torch.logit(u_mean))

    v_ctrl = unit_to_velocity(u_mean, config.v_min, config.v_max)
    print(
        f"  Initialized mean policy mu from CNN pretrain: {ckpt_path}\n"
        f"  init v_ctrl range=[{v_ctrl.min().item():.1f}, {v_ctrl.max().item():.1f}]"
    )


# ---------------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------------

def make_summary_figure(
    save_path: str,
    step: int,
    v_true: torch.Tensor,
    v_best: torch.Tensor,
    v_mean: torch.Tensor,
    p_obs: torch.Tensor,
    p_best: torch.Tensor,
    history: list[dict],
):
    """Generate 2×3 summary figure for training monitoring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_true_np = v_true.detach().cpu().numpy()
    v_best_np = v_best.detach().cpu().numpy()
    v_mean_np = v_mean.detach().cpu().numpy()
    p_obs_np = p_obs.detach().cpu().numpy()
    p_best_np = p_best.detach().cpu().numpy()

    vmin = float(v_true_np.min())
    vmax = float(v_true_np.max())
    extent_km = [0, 0.7, 0.7, 0]  # 70 × 10m = 0.7 km

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    # Row 1: Velocity models (data is [nz, nx]; imshow directly — z vertical, x horizontal)
    im0 = axes[0, 0].imshow(v_true_np, origin="upper", cmap="turbo", vmin=vmin, vmax=vmax,
                             aspect="equal", extent=extent_km)
    axes[0, 0].set_title("True Velocity")
    axes[0, 0].set_xlabel("x (km)")
    axes[0, 0].set_ylabel("Depth (km)")
    plt.colorbar(im0, ax=axes[0, 0], label="m/s")

    im1 = axes[0, 1].imshow(v_best_np, origin="upper", cmap="turbo", vmin=vmin, vmax=vmax,
                             aspect="equal", extent=extent_km)
    axes[0, 1].set_title(f"Best-of-G Velocity (step {step})")
    axes[0, 1].set_xlabel("x (km)")
    axes[0, 1].set_ylabel("Depth (km)")
    plt.colorbar(im1, ax=axes[0, 1], label="m/s")

    err = np.abs(v_best_np - v_true_np)
    im2 = axes[0, 2].imshow(err, origin="upper", cmap="hot", aspect="equal", extent=extent_km)
    axes[0, 2].set_title(f"|Best − True|  MAE={err.mean():.1f}")
    axes[0, 2].set_xlabel("x (km)")
    axes[0, 2].set_ylabel("Depth (km)")
    plt.colorbar(im2, ax=axes[0, 2], label="m/s")

    # Row 2: Shot gather comparison + reward curves + diagnostics
    shot_idx = min(0, p_obs_np.shape[0] - 1)
    # Clip to percentiles for better reflection visibility
    vmin_s = float(np.percentile(p_obs_np[shot_idx], 2))
    vmax_s = float(np.percentile(p_obs_np[shot_idx], 98))

    # Shot gather: AcquisitionForward returns [n_shots, n_receivers, nt].
    # imshow expects rows as y/time and columns as x/receiver, so transpose one shot.
    # extent: time 0→1s (1000 steps × 0.001s), receivers 0→0.7km (70 × 10m)
    axes[1, 0].imshow(p_obs_np[shot_idx].T, aspect="auto", cmap="seismic",
                       vmin=vmin_s, vmax=vmax_s, origin="upper",
                       extent=[0, 0.7, 1.0, 0])
    axes[1, 0].set_title(f"Observed Shot {shot_idx}")
    axes[1, 0].set_xlabel("Receiver x")
    axes[1, 0].set_ylabel("Time (s)")

    steps_hist = [h["step"] for h in history]
    rewards_l1 = [h.get("reward_l1_mean", 0) for h in history]
    rewards_l2 = [h.get("reward_l2_mean", 0) for h in history]
    if steps_hist:
        axes[1, 1].plot(steps_hist, rewards_l1, label="R_L1", alpha=0.8)
        axes[1, 1].plot(steps_hist, rewards_l2, label="R_L2", alpha=0.8)
        axes[1, 1].set_title("Reward Curves")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Step")

    maes = [h.get("best_mae_global", h.get("mae_best", 0)) for h in history]
    ratios = [h.get("ratio_mean", 1.0) for h in history]
    if steps_hist:
        ax_diag = axes[1, 2]
        ax_diag.plot(steps_hist, maes, "b-", label="MAE best", alpha=0.8)
        ax_diag.set_xlabel("Step")
        ax_diag.set_ylabel("MAE", color="b")
        ax_diag2 = ax_diag.twinx()
        ax_diag2.plot(steps_hist, ratios, "r--", label="Ratio mean", alpha=0.8)
        ax_diag2.set_ylabel("Ratio mean", color="r")
        ax_diag2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax_diag.set_title("MAE & Ratio")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_progression_figure(
    out_dir: str,
    v_true: torch.Tensor,
    v_init: torch.Tensor,
    v_final: torch.Tensor,
    best_mae: float,
    best_step: int,
):
    """Save init → final → true progression comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_true_np = v_true.detach().cpu().numpy()
    v_init_np = v_init.detach().cpu().numpy() if isinstance(v_init, torch.Tensor) else v_init
    v_final_np = v_final.detach().cpu().numpy() if isinstance(v_final, torch.Tensor) else v_final

    vmin = float(v_true_np.min())
    vmax = float(v_true_np.max())
    extent_km = [0, 0.7, 0.7, 0]

    init_mae = np.abs(v_init_np - v_true_np).mean()
    final_mae = np.abs(v_final_np - v_true_np).mean()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        f"Initial Best (MAE={init_mae:.0f})",
        f"Final Best @step {best_step} (MAE={final_mae:.0f})",
        "True Velocity",
    ]
    models = [v_init_np, v_final_np, v_true_np]
    for ax, title, model in zip(axes[0], titles, models):
        im = ax.imshow(model, origin="upper", cmap="turbo", vmin=vmin, vmax=vmax,
                       aspect="equal", extent=extent_km)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("Depth (km)")

    # Row 2: Error maps + profile
    titles_r2 = ["|Init − True|", "|Final − True|", "Vertical Profile"]
    errors = [np.abs(v_init_np - v_true_np), np.abs(v_final_np - v_true_np), None]
    for ax, title, err in zip(axes[1], titles_r2, errors):
        if err is None:
            x_center = v_true_np.shape[0] // 2
            ax.plot(v_true_np[x_center, :], np.arange(v_true_np.shape[1]) * 0.01,
                    "k-", label="True", linewidth=2)
            ax.plot(v_init_np[x_center, :], np.arange(v_true_np.shape[1]) * 0.01,
                    "gray", label=f"Init (MAE={init_mae:.0f})", linewidth=1, alpha=0.7)
            ax.plot(v_final_np[x_center, :], np.arange(v_true_np.shape[1]) * 0.01,
                    "b-", label=f"Final (MAE={final_mae:.0f})", linewidth=1.5)
            ax.invert_yaxis()
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Depth (km)")
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            im = ax.imshow(err, origin="upper", cmap="hot",
                           aspect="equal", extent=extent_km)
            ax.set_title(title)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("Depth (km)")
            plt.colorbar(im, ax=ax, fraction=0.046, label="Error (m/s)")

    fig.suptitle(f"Progression: Init (MAE={init_mae:.0f}) → Final (MAE={final_mae:.0f})", fontsize=13)
    fig.tight_layout()
    save_path = os.path.join(out_dir, "progression.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Decoder fine-tuning (post-RL)
# ---------------------------------------------------------------------------

def _finetune_decoder(
    vae_decoder,
    policy,
    config: TrainConfig,
    device: torch.device,
    best_payload: dict,
    out_dir: str,
    steps: int = 200,
    lr: float = 1e-4,
    reg_weight: float = 0.1,
):
    """
    Fine-tune unfrozen decoder layers after RL using prior-based loss.

    The loss encourages physically plausible velocity models without needing
    the forward simulator (differentiable prior loss + L2 regularization
    toward the frozen decoder output).
    """
    import copy
    print(f"\n{'='*60}")
    print(f"Decoder fine-tuning: {steps} steps, lr={lr}")
    print(f"{'='*60}")

    # Save frozen decoder output as reference
    vae_decoder.vae.eval()
    z_best = best_payload.get("z", None)
    if z_best is None:
        z_best = best_payload.get("v_ctrl")  # fallback
    if z_best is None:
        # Use policy mean
        z_best = policy.mu.detach()
    z_best = z_best.to(device).unsqueeze(0) if z_best.ndim == 1 else z_best.to(device)

    with torch.no_grad():
        v_frozen = vae_decoder.decode(z_best).detach().clone()

    # Setup optimizer for unfrozen decoder params only
    opt = torch.optim.AdamW(vae_decoder.unfrozen_params, lr=lr)

    v_min_t = torch.tensor(config.v_min, device=device)
    v_max_t = torch.tensor(config.v_max, device=device)

    best_loss = float("inf")
    best_state = None

    for step_i in range(1, steps + 1):
        vae_decoder.vae.train()
        # Only unfrozen params get gradients
        v_current = vae_decoder.decode_grad(z_best)

        # Prior loss
        dx = torch.diff(v_current, dim=2)
        dz = torch.diff(v_current, dim=3)
        loss_smooth = dx.pow(2).mean() + dz.pow(2).mean()
        loss_bound = (
            torch.clamp(v_min_t - v_current, min=0).pow(2).mean()
            + torch.clamp(v_current - v_max_t, min=0).pow(2).mean()
        )
        # Regularization: stay close to frozen output
        loss_reg = (v_current - v_frozen).pow(2).mean()

        loss = loss_smooth + loss_bound + reg_weight * loss_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(vae_decoder.vae.decoder.state_dict())

        if step_i % 50 == 0 or step_i == 1:
            with torch.no_grad():
                mae_current = (v_current - v_frozen).abs().mean().item()
            print(f"  ft step {step_i:4d}/{steps} | loss={loss.item():.6f} "
                  f"smooth={loss_smooth.item():.4f} bound={loss_bound.item():.4f} "
                  f"reg={loss_reg.item():.4f} ΔMAE={mae_current:.2f}")

    # Restore best state
    if best_state is not None:
        vae_decoder.vae.decoder.load_state_dict(best_state, strict=False)

    # Final decode with fine-tuned decoder
    vae_decoder.vae.eval()
    with torch.no_grad():
        v_final = vae_decoder.decode(z_best).squeeze().cpu().numpy()
    np.save(os.path.join(out_dir, "best_velocity_ft.npy"), v_final)
    print(f"  Fine-tuned model saved to best_velocity_ft.npy")
    print(f"{'='*60}")


def _update_decoder(vae_decoder, z_best, device, config, decoder_lr=5e-4, reg_weight=0.1):
    """One-step decoder update using prior loss on the best latent vector."""
    vae_decoder.vae.train()
    v = vae_decoder.decode_grad(z_best.to(device))
    dx = torch.diff(v, dim=2)
    dz = torch.diff(v, dim=3)
    loss_smooth = dx.pow(2).mean() + dz.pow(2).mean()
    v_min_t = torch.tensor(config.v_min, device=device)
    v_max_t = torch.tensor(config.v_max, device=device)
    loss_bound = (torch.clamp(v_min_t - v, min=0).pow(2).mean() +
                  torch.clamp(v - v_max_t, min=0).pow(2).mean())
    with torch.no_grad():
        v_frozen = vae_decoder.decode(z_best.to(device)).detach()
    loss_reg = (v - v_frozen).pow(2).mean()
    loss = loss_smooth + loss_bound + reg_weight * loss_reg

    # Only update unfrozen decoder params
    if vae_decoder.unfrozen_params:
        opt = torch.optim.AdamW(vae_decoder.unfrozen_params, lr=decoder_lr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    vae_decoder.vae.eval()


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and device.type == "cpu":
        print("⚠️  CUDA not available, falling back to CPU")
    print(f"Using device: {device}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ---- Setup geometry & forward simulator ----
    geom = AcquisitionGeometry(
        nx_model=config.nx_model,
        nz_model=config.nz_model,
        dx=config.dx,
        dt=config.dt,
        freq=config.freq,
        nt=config.nt,
        n_shots=config.n_shots,
        n_receivers=config.n_receivers,
        pml_width=config.pml_width,
        geometry=config.geometry,
    )
    if not geom.validate_receiver_z():
        raise RuntimeError(
            f"Receivers at z_padded={geom.pml_width + geom.nz_model - 1} fall inside PML! "
            f"PML starts at z={geom.pml_width + geom.nz_model}. "
            f"Decrease nz_model or increase pml_width."
        )

    forward = AcquisitionForward(geom)
    reconstructor = VelocityReconstructor(
        nx_model=config.nx_model,
        nz_model=config.nz_model,
    )

    # ---- Load or generate observation data ----
    print(f"Loading model from: {config.model_source}")
    if config.model_source == "cva":
        from utils.data_loader import CurveVelADataset
        cva_base = CurveVelADataset(config.cva_root)
        item = cva_base.get_by_file_index(config.cva_file_idx, config.cva_sample_idx)
        v_true_np = item["model"].astype(np.float32)
        if v_true_np.shape != (config.nx_model, config.nz_model):
            raise ValueError(
                f"CVA velocity shape {v_true_np.shape} != "
                f"({config.nx_model}, {config.nz_model})"
            )
        p_data, v_true = make_observation_from_velocity(v_true_np, geom, str(device))
        print(f"  Loaded CVA[{config.cva_file_idx}][{config.cva_sample_idx}], "
              f"v range=[{v_true_np.min():.0f}, {v_true_np.max():.0f}]")
    elif config.model_source == "fva":
        # FVA: load velocity model, synthesize seismic via deepwave
        import glob, re
        fva_files = sorted(
            glob.glob(os.path.join(config.fva_root, "model*.npy")),
            key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
        )
        if config.cva_file_idx >= len(fva_files):
            raise ValueError(f"FVA file_idx {config.cva_file_idx} >= {len(fva_files)} files")
        fva_data = np.load(fva_files[config.cva_file_idx], mmap_mode="r")
        v_true_np = np.asarray(fva_data[config.cva_sample_idx], dtype=np.float32).copy()
        if v_true_np.ndim == 3 and v_true_np.shape[0] == 1:
            v_true_np = v_true_np[0]
        if v_true_np.shape != (config.nx_model, config.nz_model):
            raise ValueError(
                f"FVA velocity shape {v_true_np.shape} != "
                f"({config.nx_model}, {config.nz_model})"
            )
        p_data, v_true = make_observation_from_velocity(v_true_np, geom, str(device))
        print(f"  Loaded FVA[{config.cva_file_idx}][{config.cva_sample_idx}], "
              f"v range=[{v_true_np.min():.0f}, {v_true_np.max():.0f}] "
              f"(seismic synthesized)")
    elif config.model_source == "smooth":
        # B-spline smoothed CVA model
        path = os.path.join(config.smooth_root, f"smooth_cva{config.cva_file_idx}.npy")
        v_true_np = np.load(path).astype(np.float32)
        if v_true_np.shape != (config.nx_model, config.nz_model):
            raise ValueError(f"Smooth model shape {v_true_np.shape} != ({config.nx_model},{config.nz_model})")
        p_data, v_true = make_observation_from_velocity(v_true_np, geom, str(device))
        print(f"  Loaded smooth CVA[{config.cva_file_idx}], "
              f"v range=[{v_true_np.min():.0f}, {v_true_np.max():.0f}]")
    else:
        v_true_np = make_synthetic_layered_model(
            config.nx_model, config.nz_model, config.v_min, config.v_max
        )
        p_data, v_true = make_observation_from_velocity(v_true_np, geom, str(device))
        print(f"  Created synthetic layered model, "
              f"v range=[{v_true_np.min():.0f}, {v_true_np.max():.0f}]")

    assert_shot_receiver_time(
        p_data,
        n_shots=config.n_shots,
        n_receivers=config.n_receivers,
        nt=config.nt,
        name="p_data",
    )
    print(f"  p_data shape: {tuple(p_data.shape)}")
    print(f"  v_true shape: {tuple(v_true.shape)}")

    # ---- Initialize policy ----
    if config.policy_type == "cnn":
        policy = CNNBetaMeanPolicy(
            nx_ctrl=config.nx_ctrl,
            nz_ctrl=config.nz_ctrl,
            in_channels=config.n_shots,
            embed_dim=config.cnn_embed_dim,
            v_min=config.v_min,
            v_max=config.v_max,
            init_kappa=config.init_kappa,
        ).to(device)
        if config.pretrain_ckpt:
            ckpt = torch.load(config.pretrain_ckpt, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            policy.load_pretrained_velocity_predictor(state_dict)
            print(f"  Loaded CNN pretrain checkpoint: {config.pretrain_ckpt}")
        p_data_input = p_data.unsqueeze(0).to(device)
    elif config.policy_type == "mean":
        policy = LearnableBetaMeanPolicy(
            nx_ctrl=config.nx_ctrl,
            nz_ctrl=config.nz_ctrl,
            v_min=config.v_min,
            v_max=config.v_max,
            init_kappa=config.init_kappa,
        ).to(device)
        if config.pretrain_ckpt:
            init_mean_policy_from_cnn_pretrain(
                policy=policy,
                ckpt_path=config.pretrain_ckpt,
                p_data=p_data,
                config=config,
                device=device,
            )
        p_data_input = None
    elif config.policy_type == "gaussian":
        policy = GaussianMeanPolicy(
            nx_ctrl=config.nx_ctrl,
            nz_ctrl=config.nz_ctrl,
            v_min=config.v_min,
            v_max=config.v_max,
        ).to(device)
        p_data_input = None
    elif config.policy_type == "latent":
        # Latent-space Gaussian policy (Phase III)
        from agents.latent_policy import LearnableLatentPolicy, VAEDecoder
        if config.vae_ckpt is None:
            raise ValueError("--vae_ckpt required for --policy_type latent")
        vae_decoder = VAEDecoder(config.vae_ckpt, device=str(device),
                                 unfreeze_last_n=config.unfreeze_decoder)
        policy = LearnableLatentPolicy(
            latent_dim=config.latent_dim,
            init_sigma=0.5,
        ).to(device)
        p_data_input = None
        print(f"  VAE decoder loaded: latent_dim={vae_decoder.latent_dim}")
        print(f"  Velocity range: [{vae_decoder.v_min:.0f}, {vae_decoder.v_max:.0f}] m/s")
    else:  # "learnable" legacy
        policy = LearnableBetaSplinePolicy(
            nx_ctrl=config.nx_ctrl,
            nz_ctrl=config.nz_ctrl,
            v_min=config.v_min,
            v_max=config.v_max,
            init_alpha=config.init_alpha,
            init_beta=config.init_beta,
        ).to(device)
        p_data_input = None

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy: {config.policy_type}, {n_params:,} parameters")

    # Resume from checkpoint (progressive training)
    if config.resume_ckpt:
        ckpt = torch.load(config.resume_ckpt, map_location=device)
        policy.load_state_dict(ckpt)
        print(f"  Loaded policy from: {config.resume_ckpt}")

    # ---- Setup optimizer & GRPO-Guard ----
    params = list(policy.parameters())
    if config.optimizer == "muon":
        from agents.muon import MuonState, muon_step
        muon_state = MuonState()
        optimizer = None  # manual step
        print(f"  Optimizer: Muon (momentum={config.muon_momentum}, ns_steps={config.muon_ns_steps})")
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        print(f"  Optimizer: AdamW (lr={config.lr})")
    guard_state = RatioGuardState(
        correction_enabled=config.guard_enabled,
        correction_threshold=config.guard_threshold,
    ) if config.guard_enabled else None

    reward_weights = RewardWeights(
        l1=config.reward_l1_weight,
        l2=config.reward_l2_weight,
        si=config.reward_si_weight,
        prior=config.reward_prior_weight,
        tt=config.reward_tt_weight,
        fwi2=config.fwi_weight2,
    )

    # ---- Training state ----
    os.makedirs(config.out_dir, exist_ok=True)
    np.save(os.path.join(config.out_dir, "true_velocity.npy"), v_true.detach().cpu().numpy())

    # Save config
    config_path = os.path.join(config.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({k: str(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in vars(config).items()}, f, indent=2, default=str)

    metrics_path = os.path.join(config.out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("step,reward_l1_mean,reward_l2_mean,reward_tt_mean,reward_prior_mean,"
                "mae_display,mae_oracle_best,best_mae_global,"
                "ratio_mean,ratio_std,clip_frac,temperature,entropy,wall_time\n")

    best_mae = float("inf")
    best_mae_step = 0
    best_mae_payload: dict[str, torch.Tensor | float | int] | None = None
    v_init_best: torch.Tensor | None = None  # save initial best for progression viz
    history: list[dict] = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"Starting training: {config.steps} steps, G={config.group_size}")
    print(f"Rewards: L1×{config.reward_l1_weight}, L2×{config.reward_l2_weight}, SI×{config.reward_si_weight}, Prior×{config.reward_prior_weight}")
    print(f"Best criterion: {config.best_criterion}, Geometry: {config.geometry}")
    print(f"Temperature: {config.init_temperature} → {config.final_temperature}")
    print(f"{'='*60}\n")

    # ---- Main loop ----
    for step in range(1, config.steps + 1):
        # Compute temperature
        anneal_frac = min(1.0, step / max(1, config.anneal_steps))
        temperature = config.init_temperature + anneal_frac * (
            config.final_temperature - config.init_temperature
        )

        # ---- Phase 1: Rollout with current policy ----
        policy.eval()
        with torch.no_grad():
            if config.policy_type == "latent":
                # Sample from latent Gaussian policy
                z_samples = policy.sample(config.group_size)  # [G, latent_dim]
                logp_old = policy.log_prob(z_samples)  # [G, latent_dim]
                # Decode latent vectors to velocity models
                v_models = vae_decoder.decode(z_samples).squeeze(1)  # [G, nx, nz]
            else:
                sample_out = policy.sample(p_data_input, n=config.group_size, temperature=temperature)
                v_ctrl = sample_out["velocity"]  # [G, 1, nx_ctrl, nz_ctrl]
                logp_old = sample_out["log_prob"]  # [G, 1, nx_ctrl, nz_ctrl]

                # Squeeze batch dim (B=1 for single observation)
                v_ctrl_sq = v_ctrl.squeeze(1)  # [G, nx_ctrl, nz_ctrl]

                # Reconstruct velocity models
                v_models = []
                for i in range(config.group_size):
                    vm = reconstructor.reconstruct(v_ctrl_sq[i])
                    v_models.append(vm.squeeze(0))  # [nx, nz]
                v_models = torch.stack(v_models, dim=0)  # [G, nx, nz]

        # Forward simulation (serial — deepwave limitation)
        p_pred = forward.simulate_batch(v_models, device=str(device))  # [G, n_shots, n_receivers, nt]
        assert_batch_shot_receiver_time(
            p_pred,
            group_size=config.group_size,
            n_shots=config.n_shots,
            n_receivers=config.n_receivers,
            nt=config.nt,
            name="p_pred",
        )

        # Apply lowpass filter if freq_band specified
        if config.freq_band:
            cutoff = float(config.freq_band.split("-")[-1])
            from agents.fwi_rewards import lowpass_filter
            p_pred_f = lowpass_filter(p_pred, cutoff, config.dt)
            p_data_f = lowpass_filter(p_data.unsqueeze(0), cutoff, config.dt).squeeze(0)
        else:
            p_pred_f = p_pred
            p_data_f = p_data

        # Compute rewards (use filtered data if freq_band set)
        reward_dict = compute_group_rewards(p_pred_f, p_data_f, reward_weights)
        r_l1 = reward_dict.get("l1", torch.zeros(config.group_size, device=device))
        r_l2 = reward_dict.get("l2", torch.zeros(config.group_size, device=device))

        # Override L2 with FWI variant if specified
        if config.reward_l2_weight > 0 and config.fwi_type != "l2":
            from agents.fwi_rewards import compute_fwi_reward
            fwi_kwargs = {}
            if config.fwi_type == "wasserstein":
                fwi_kwargs["normalize"] = config.wasserstein_normalize
            elif config.fwi_type == "wasserstein_w2":
                pass  # uses defaults (eta=1e-3)
            elif config.fwi_type == "ncc_maxlag":
                fwi_kwargs["lag_max"] = config.ncc_lag_max
                fwi_kwargs["lag_penalty"] = config.ncc_lag_penalty
            elif config.fwi_type == "awi":
                fwi_kwargs["version"] = config.awi_version
            r_l2 = compute_fwi_reward(p_pred_f, p_data_f, fwi_type=config.fwi_type, **fwi_kwargs)

        # Second FWI reward for multi-FWI mixing (Phase 5B)
        r_fwi2 = torch.zeros(config.group_size, device=device)
        if config.fwi_type2 and config.fwi_weight2 > 0:
            from agents.fwi_rewards import compute_fwi_reward as compute_fwi2
            fwi2_kwargs = {}
            if config.fwi_type2 == "wasserstein":
                fwi2_kwargs["normalize"] = config.wasserstein_normalize
            elif config.fwi_type2 == "wasserstein_w2":
                pass  # defaults
            elif config.fwi_type2 == "ncc_maxlag":
                fwi2_kwargs["lag_max"] = config.ncc_lag_max
                fwi2_kwargs["lag_penalty"] = config.ncc_lag_penalty
            elif config.fwi_type2 == "awi":
                fwi2_kwargs["version"] = config.awi_version
            r_fwi2 = compute_fwi2(p_pred_f, p_data_f, fwi_type=config.fwi_type2, **fwi2_kwargs)

        # Travel-time reward (if enabled) — TT uses raw data (timing sensitive)
        r_tt = torch.zeros(config.group_size, device=device)
        if config.reward_tt_weight > 0:
            if config.reward_tt_log:
                from agents.traveltime_reward import traveltime_reward_log
                r_tt = traveltime_reward_log(p_pred, p_data)
            else:
                from agents.traveltime_reward import traveltime_reward
                r_tt = traveltime_reward(p_pred, p_data)

        # Prior reward (smoothness + monotonicity + bounds)
        from agents.rl_objectives import velocity_prior_reward
        r_prior = velocity_prior_reward(
            v_models,
            v_min=config.v_min,
            v_max=config.v_max,
            smooth_weight=1.0,
            monotonic_weight=0.1,
            bound_weight=1.0,
        )  # [G]

        # SI reward (RTM imaging energy) — compute every si_every steps
        r_si = torch.zeros(config.group_size, device=device)
        if config.reward_si_weight > 0 and step % config.si_every == 0:
            try:
                from example_to_dyy.funcs.RTM import rtm_imaging_batch_all_forw
                import deepwave
                nbc = config.pml_width
                src_loc = torch.zeros(config.n_shots, 1, 2, dtype=torch.long, device=device)
                src_x = torch.linspace(0, config.nx_model-1, config.n_shots).long().to(device)
                src_loc[:, 0, 0] = src_x
                src_loc[:, 0, 1] = 0
                rec_loc = torch.zeros(config.n_shots, config.n_receivers, 2, dtype=torch.long, device=device)
                rec_loc[:, :, 0] = torch.arange(config.n_receivers, device=device).unsqueeze(0).repeat(config.n_shots, 1)
                rec_loc[:, :, 1] = 0 if config.geometry == "reflection" else config.nz_model - 1
                peak_time = 1.5 / config.freq
                src_amp = deepwave.wavelets.ricker(config.freq, config.nt, config.dt, peak_time).repeat(config.n_shots, 1, 1).to(device)
                for g_idx in range(config.group_size):
                    image = torch.zeros_like(v_models[g_idx])
                    h = rtm_imaging_batch_all_forw(
                        image=image,
                        v_apply=v_models[g_idx].to(device),
                        grid_spacing=[config.dx, config.dx],
                        dt=config.dt,
                        source_amplitudes=src_amp,
                        source_locations=src_loc,
                        obsv_data_masked=p_data.permute(0, 2, 1).to(device),  # [n_shots, n_receivers, nt]
                        receiver_locations=rec_loc,
                        batch_size=config.n_shots,
                        pml_width=[nbc]*4,
                        pml_freq=config.freq,
                        callback_freq=1,
                        illum=False,
                        outSI=True,
                    )
                    r_si[g_idx] = float(h)
            except Exception as e:
                if step == config.si_every:
                    print(f"  [WARN] SI reward unavailable: {e}")

        # GDPO advantage
        adv = gdpo_advantage(
            {"l1": r_l1.unsqueeze(1), "l2": r_l2.unsqueeze(1),
             "tt": r_tt.unsqueeze(1), "si": r_si.unsqueeze(1), "prior": r_prior.unsqueeze(1),
             "fwi2": r_fwi2.unsqueeze(1)},
            reward_weights,
            batch_norm=False,
        )  # [G, 1]

        # Compute best-of-G stats
        total_reward = r_l1 + r_l2
        mae_per_sample = (v_models - v_true.unsqueeze(0)).abs().mean(dim=(1, 2))
        oracle_best_idx = int(mae_per_sample.argmin().item())
        v_oracle_best = v_models[oracle_best_idx]
        mae_display = float(mae_per_sample[oracle_best_idx].detach().cpu().item())
        mae_oracle_best = float(mae_per_sample[oracle_best_idx].detach().cpu().item())

        # Select display best by configured criterion
        if config.best_criterion == "si":
            display_best_idx = int(r_si.argmax().item())
        elif config.best_criterion == "l2":
            display_best_idx = int(r_l2.argmax().item())
        else:  # mae or default
            display_best_idx = oracle_best_idx
        v_display_best = v_models[display_best_idx]
        mae_display = float(mae_per_sample[display_best_idx].detach().cpu().item())

        # Track best by MAE (always)
        if mae_oracle_best < best_mae:
            best_mae = mae_oracle_best
            best_mae_step = step
            best_mae_payload = {
                "step": step,
                "sample_idx": oracle_best_idx,
                "mae": best_mae,
                "v_model": v_oracle_best.detach().cpu(),
                "v_ctrl": z_samples[oracle_best_idx].detach().cpu() if config.policy_type == "latent" else v_ctrl_sq[oracle_best_idx].detach().cpu(),
                "p_pred": p_pred[oracle_best_idx].detach().cpu(),
                "r_l1": float(r_l1[oracle_best_idx].detach().cpu().item()),
                "r_l2": float(r_l2[oracle_best_idx].detach().cpu().item()),
                "r_prior": float(r_prior[oracle_best_idx].detach().cpu().item()),
                "r_si": float(r_si[oracle_best_idx].detach().cpu().item()),
                "total_reward": float(total_reward[oracle_best_idx].detach().cpu().item()),
                "temperature": float(temperature),
            }
            ckpt_best_path = os.path.join(config.out_dir, "policy_best.pt")
            torch.save(policy.state_dict(), ckpt_best_path)
            torch.save(best_mae_payload, os.path.join(config.out_dir, "best_payload.pt"))

        # ---- Phase 2-5: PPO multi-epoch update ----
        policy.train()
        total_loss = 0.0
        for epoch_i in range(config.ppo_epochs):
            # Recompute log-prob under current (updating) policy
            if config.policy_type == "cnn":
                logp_new = policy.log_prob(p_data_input, sample_out["u"], temperature=temperature)
            elif config.policy_type == "latent":
                logp_new = policy.log_prob(z_samples)  # [G, latent_dim]
                # Reshape to [G, 1, latent_dim, 1] for clipped_policy_loss
                logp_new = logp_new.unsqueeze(1).unsqueeze(-1)  # [G, 1, latent_dim, 1]
                logp_old_reshaped = logp_old.unsqueeze(1).unsqueeze(-1)  # same
            else:
                logp_new = policy.log_prob(None, sample_out["u"], temperature=temperature)
            # PPO/GDPO clipped policy gradient with GRPO-Guard ratio correction.
            # log_prob tensors are [G, B=1, nx_ctrl, nz_ctrl], advantage is [G, B=1].
            if config.policy_type == "latent":
                _logp_new = logp_new  # already [G, 1, latent_dim, 1]
                _logp_old = logp_old_reshaped  # [G, 1, latent_dim, 1]
            else:
                _logp_new = logp_new
                _logp_old = logp_old
            loss, ratio_stats = clipped_policy_loss(
                logp_new=_logp_new,
                logp_old=_logp_old,
                advantages=adv,
                epsilon_low=config.epsilon_low,
                epsilon_high=config.epsilon_high,
                guard_state=guard_state,
                token_mean=False,
            )

            # Entropy bonus: prevent distribution collapse
            if config.policy_type in ("learnable", "mean", "gaussian"):
                ent_tensor = policy.raw_entropy()
            elif config.policy_type == "latent":
                ent_tensor = policy.entropy().mean()  # per-dim mean entropy
            else:
                alpha_p, beta_p = policy.forward(p_data_input)
                from agents.beta_policy import beta_entropy
                ent_tensor = beta_entropy(alpha_p, beta_p)
            loss = loss - config.entropy_bonus * ent_tensor
            ratio_stats["loss"] = float(loss.detach().cpu().item())

            if config.optimizer == "muon":
                policy.zero_grad()
                loss.backward()
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
                muon_step(params, config.lr, muon_state, momentum=config.muon_momentum,
                          ns_steps=config.muon_ns_steps, weight_decay=config.weight_decay)
            else:
                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())

        avg_loss = total_loss / config.ppo_epochs

        # ---- Decoder update (if unfrozen) ----
        if config.policy_type == "latent" and config.unfreeze_decoder > 0 and oracle_best_idx < len(z_samples):
            _update_decoder(vae_decoder, z_samples[oracle_best_idx:oracle_best_idx+1],
                          device, config, decoder_lr=config.lr * 0.1)

        # Per-component means for logging
        r_l1_mean = float(r_l1.mean().item())
        r_l2_mean = float(r_l2.mean().item())
        r_tt_mean = float(r_tt.mean().item())
        r_prior_mean = float(r_prior.mean().item())
        r_fwi2_mean = float(r_fwi2.mean().item()) if config.fwi_type2 else 0.0
        r_si_mean = float(r_si.mean().detach().cpu().item())

        # Compute entropy
        if config.policy_type in ("learnable", "mean", "gaussian"):
            ent = float(policy.entropy.detach().cpu().item())
        elif config.policy_type == "latent":
            ent = float(policy.entropy().mean().detach().cpu().item())
        else:
            alpha, beta_param = policy.forward(p_data_input)
            from agents.beta_policy import beta_entropy
            ent = float(beta_entropy(alpha, beta_param).detach().cpu().item())

        wall_time = time.time() - t_start

        # Save initial best velocity for progression visualization
        if step == 1 and v_init_best is None:
            v_init_best = v_oracle_best.detach().cpu()
            np.save(os.path.join(config.out_dir, "init_velocity.npy"), v_init_best.numpy())

        log_entry = {
            "step": step,
            "reward_l1_mean": r_l1_mean,
            "reward_l2_mean": r_l2_mean,
            "reward_prior_mean": r_prior_mean,
            "mae_display": mae_display,
            "mae_oracle_best": mae_oracle_best,
            "best_mae_global": best_mae,
            "temperature": temperature,
            "entropy": ent,
        }
        log_entry.update(ratio_stats)
        history.append(log_entry)

        with open(metrics_path, "a") as f:
            f.write(f"{step},{r_l1_mean:.2f},{r_l2_mean:.2f},{r_tt_mean:.4f},{r_prior_mean:.4f},"
                    f"{mae_display:.2f},{mae_oracle_best:.2f},{best_mae:.2f},"
                    f"{ratio_stats['ratio_mean']:.6f},{ratio_stats['ratio_std']:.6f},"
                    f"{ratio_stats['clip_fraction']:.6f},{temperature:.4f},{ent:.4f},"
                    f"{wall_time:.1f}\n")

        # Build log line with fwi2 if active
        log_parts = [f"step {step:4d}/{config.steps} | ",
                     f"R_L1={r_l1_mean:.1f} R_L2={r_l2_mean:.1f} "]
        if config.fwi_type2:
            log_parts.append(f"R_F2={r_fwi2_mean:.1f} ")
        log_parts.append(f"R_TT={r_tt_mean:.3f} R_P={r_prior_mean:.3f} | ")
        log_parts.append(f"MAE_reward={mae_display:.1f} MAE_oracle={mae_oracle_best:.1f} "
                         f"(global={best_mae:.1f}@{best_mae_step}) | "
                         f"ratio={ratio_stats['ratio_mean']:.4f} "
                         f"clip={ratio_stats['clip_fraction']:.4f} | "
                         f"ent={ent:.4f} T={temperature:.3f} | "
                         f"loss={avg_loss:.4f}")
        log_line = "".join(log_parts)
        print(log_line)

        # Reward hacking detection: check if one reward component dominates
        if config.fwi_type2 and step > 100 and step % 100 == 0:
            r_l2_std = float(r_l2.std().item())
            r_fwi2_std = float(r_fwi2.std().item())
            # If one component has near-zero variance, it's being ignored
            if r_l2_std < 1e-3 or r_fwi2_std < 1e-3:
                print(f"  ⚠️  Reward hacking risk: L2_std={r_l2_std:.4f}, F2_std={r_fwi2_std:.4f}")

        # ---- Early stopping ----
        if step >= config.early_stop_window and best_mae_step > 0:
            steps_since_best = step - best_mae_step
            if steps_since_best >= config.early_stop_patience:
                print(f"  Early stop: no MAE improvement for {steps_since_best} steps "
                      f"(best={best_mae:.1f} @ step {best_mae_step})")
                break

        # ---- Save checkpoint ----
        if step % config.save_every == 0 or step == config.steps:
            # Save model
            ckpt_path = os.path.join(config.out_dir, f"policy_step_{step:06d}.pt")
            torch.save(policy.state_dict(), ckpt_path)

            # Make summary figure
            # Compute posterior mean velocity
            if config.policy_type in ("learnable", "mean"):
                alpha_p, beta_p = policy.forward()
                # Mean of Beta: α/(α+β)
                u_mean = alpha_p / (alpha_p + beta_p)
            elif config.policy_type == "gaussian":
                # Mean of Gaussian in logit space → sigmoid → unit
                u_mean = torch.sigmoid(policy.mu).unsqueeze(0)  # [1, H, W]
            elif config.policy_type == "latent":
                z_mean = policy.mu.unsqueeze(0)  # [1, latent_dim]
                v_mean = vae_decoder.decode(z_mean).squeeze(0).squeeze(0)  # [nx, nz]
            else:
                alpha_p, beta_p = policy.forward(p_data_input)
                u_mean = alpha_p / (alpha_p + beta_p)
            if config.policy_type != "latent":
                v_ctrl_mean = unit_to_velocity(
                    u_mean.squeeze(0), config.v_min, config.v_max
                )
                v_mean = reconstructor.reconstruct(v_ctrl_mean).squeeze(0)  # [nx, nz]

            summary_path = os.path.join(config.out_dir, f"summary_step_{step:06d}.png")
            make_summary_figure(
                save_path=summary_path,
                step=step,
                v_true=v_true,
                v_best=v_display_best,
                v_mean=v_mean,
                p_obs=p_data,
                p_best=p_pred[display_best_idx] if config.group_size > 0 else p_pred[0],
                history=history,
            )

            if best_mae_payload is not None:
                best_pt_path = os.path.join(config.out_dir, "best_by_mae.pt")
                torch.save(best_mae_payload, best_pt_path)
                np.save(
                    os.path.join(config.out_dir, "best_velocity.npy"),
                    best_mae_payload["v_model"].numpy(),
                )
                np.save(
                    os.path.join(config.out_dir, "best_ctrl.npy"),
                    best_mae_payload["v_ctrl"].numpy(),
                )
                best_summary_path = os.path.join(config.out_dir, "best_by_mae_summary.png")
                make_summary_figure(
                    save_path=best_summary_path,
                    step=int(best_mae_payload["step"]),
                    v_true=v_true,
                    v_best=best_mae_payload["v_model"].to(device=v_true.device),
                    v_mean=v_mean,
                    p_obs=p_data,
                    p_best=best_mae_payload["p_pred"].to(device=p_data.device),
                    history=history,
                )

    # ---- Final ----
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {config.steps} steps in {elapsed:.1f}s")
    print(f"Best oracle MAE: {best_mae:.1f} @ step {best_mae_step}")
    print(f"Output: {config.out_dir}")
    print(f"{'='*60}")

    # Save progression figure: init → final best → true
    if v_init_best is not None and best_mae_payload is not None:
        try:
            _save_progression_figure(
                config.out_dir, v_true, v_init_best,
                best_mae_payload["v_model"], best_mae, best_mae_step,
            )
        except Exception as e:
            print(f"  [WARN] progression figure failed: {e}")

    # Save final policy
    final_path = os.path.join(config.out_dir, "policy_final.pt")
    torch.save(policy.state_dict(), final_path)

    # Save final converged model for post-run visualization.
    with torch.no_grad():
        if config.policy_type in ("learnable", "mean"):
            alpha_p, beta_p = policy.forward()
            u_mean = alpha_p / (alpha_p + beta_p)
            v_ctrl_final = unit_to_velocity(u_mean.squeeze(0), config.v_min, config.v_max)
            v_final = reconstructor.reconstruct(v_ctrl_final).squeeze(0)
        elif config.policy_type == "gaussian":
            u_mean = torch.sigmoid(policy.mu).unsqueeze(0)
            v_ctrl_final = unit_to_velocity(u_mean.squeeze(0), config.v_min, config.v_max)
            v_final = reconstructor.reconstruct(v_ctrl_final).squeeze(0)
        elif config.policy_type == "latent":
            z_mean = policy.mu.unsqueeze(0)
            v_final = vae_decoder.decode(z_mean).squeeze(0).squeeze(0)
        else:
            alpha_p, beta_p = policy.forward(p_data_input)
            u_mean = alpha_p / (alpha_p + beta_p)
            v_ctrl_final = unit_to_velocity(u_mean.squeeze(0), config.v_min, config.v_max)
            v_final = reconstructor.reconstruct(v_ctrl_final).squeeze(0)
    np.save(os.path.join(config.out_dir, "final_velocity.npy"), v_final.detach().cpu().numpy())

    # ---- Decoder fine-tuning (if unfrozen) ----
    if config.policy_type == "latent" and config.unfreeze_decoder > 0 and best_mae_payload is not None:
        _finetune_decoder(
            vae_decoder, policy, config, device,
            best_mae_payload, out_dir=config.out_dir,
        )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase II: Multi-objective RL-FWI (Beta policy + GDPO-Guard)"
    )
    # Model
    parser.add_argument("--nx_model", type=int, default=70)
    parser.add_argument("--nz_model", type=int, default=70)
    parser.add_argument("--nx_ctrl", type=int, default=4)
    parser.add_argument("--nz_ctrl", type=int, default=4)
    parser.add_argument("--v_min", type=float, default=1500.0)
    parser.add_argument("--v_max", type=float, default=4500.0)

    # Acquisition
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--n_receivers", type=int, default=70)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--pml_width", type=int, default=40)
    parser.add_argument("--freq", type=float, default=15.0, help="Source frequency (Hz)")
    parser.add_argument("--freq_band", type=str, default="", help="Lowpass cutoff before reward, e.g. '0-5' or '0-10'")

    # Policy
    parser.add_argument("--policy_type", choices=["mean", "learnable", "cnn", "latent", "gaussian"], default="mean")
    parser.add_argument("--cnn_embed_dim", type=int, default=128)
    parser.add_argument("--init_alpha", type=float, default=2.0)
    parser.add_argument("--init_beta", type=float, default=2.0)
    parser.add_argument("--init_kappa", type=float, default=4.0)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--pretrain_mu_clip", type=float, default=0.05)
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume policy from checkpoint (progressive training)")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to trained VAE checkpoint (for --policy_type latent)")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--unfreeze_decoder", type=int, default=0, help="Unfreeze last N decoder ConvTranspose layers for RL fine-tuning")

    # RL
    parser.add_argument("--group_size", "-G", type=int, default=8)
    parser.add_argument("--steps", "-N", type=int, default=500)
    parser.add_argument("--ppo_epochs", "-K", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Clipping
    parser.add_argument("--epsilon_low", type=float, default=0.20)
    parser.add_argument("--epsilon_high", type=float, default=0.27)

    # Reward
    parser.add_argument("--reward_l1_weight", type=float, default=1.0)
    parser.add_argument("--reward_l2_weight", type=float, default=1.0)
    parser.add_argument("--fwi_type", choices=["l2", "envelope", "windowed_l2", "wasserstein", "wasserstein_w2", "contrastive",
                                                "ncc_zero", "ncc_maxlag", "envelope_ncc", "awi", "phase_func"], default="l2",
                        help="FWI data-misfit reward type")
    parser.add_argument("--fwi_type2", type=str, default="",
                        help="Second FWI reward for multi-FWI mixing (e.g. contrastive)")
    parser.add_argument("--fwi_weight2", type=float, default=0.0,
                        help="Weight for second FWI reward in GDPO (0 = disabled)")
    parser.add_argument("--wasserstein_normalize", choices=["abs", "square", "envelope"], default="abs",
                        help="Normalization for wasserstein reward: abs, square, or envelope")
    parser.add_argument("--ncc_lag_max", type=int, default=80,
                        help="Max lag (samples) for ncc_maxlag reward")
    parser.add_argument("--ncc_lag_penalty", type=float, default=0.005,
                        help="Lag penalty weight for ncc_maxlag reward")
    parser.add_argument("--awi_version", choices=["l1", "full"], default="l1",
                        help="AWI version: l1 (simple) or full (center+spread)")
    parser.add_argument("--reward_si_weight", type=float, default=0.0)
    parser.add_argument("--reward_prior_weight", type=float, default=0.05)
    parser.add_argument("--si_every", type=int, default=10)
    parser.add_argument("--best_criterion", choices=["mae", "l2", "si"], default="l2")
    parser.add_argument("--geometry", choices=["reflection", "transmission"], default="reflection")

    # Temperature
    parser.add_argument("--init_temperature", type=float, default=2.0)
    parser.add_argument("--final_temperature", type=float, default=0.1)
    parser.add_argument("--anneal_steps", type=int, default=400)

    # Entropy
    parser.add_argument("--entropy_bonus", type=float, default=0.02)

    # GRPO-Guard
    parser.add_argument("--guard_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guard_threshold", type=float, default=0.05)

    # Data
    parser.add_argument("--model_source", choices=["synthetic", "cva", "fva", "smooth"], default="synthetic")
    parser.add_argument("--cva_root", type=str, default="data/CVA/CurveVel_A")
    parser.add_argument("--cva_file_idx", type=int, default=0)
    parser.add_argument("--cva_sample_idx", type=int, default=0)
    parser.add_argument("--fva_root", type=str, default="data/FVA_model")
    parser.add_argument("--smooth_root", type=str, default="data/smooth_models")

    # Reward
    parser.add_argument("--reward_tt_weight", type=float, default=0.0, help="Travel-time reward weight")
    parser.add_argument("--reward_tt_log", action="store_true", default=False, help="Use log-scaled tt reward (amplifies small |Δt|)")
    parser.add_argument("--out_dir", type=str, default="runs/phase2")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=500)
    parser.add_argument("--early_stop_window", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainConfig(**{k: v for k, v in vars(args).items()})
    train(config)


if __name__ == "__main__":
    main()
