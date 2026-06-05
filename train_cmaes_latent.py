"""
CMA-ES baseline for latent-space RL seismic inversion.

Compares CMA-ES (evolution strategy, black-box) against PPO (policy gradient)
in the same 64D latent space with the same frozen VAE decoder.

Usage:
    python train_cmaes_latent.py --vae_ckpt runs/vae_joint_cva_fva/vae_best.pt \
        --model_source cva --cva_file_idx 50 --device cuda:3
"""

from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
import cma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.latent_policy import VAEDecoder
from agents.transmission_forward import AcquisitionGeometry, AcquisitionForward
from agents.rl_objectives import reward_l1, reward_l2, sign_preserving_log


def make_observation(v_true_np, geom, device):
    forward = AcquisitionForward(geom)
    v_t = torch.from_numpy(v_true_np).to(device=device, dtype=torch.float32)
    return forward.simulate(v_t, device=device), v_t


def evaluate_z(
    z: np.ndarray,
    vae_decoder: VAEDecoder,
    p_obs: torch.Tensor,
    device: str,
    prior_weight: float = 0.05,
    v_min: float = 1500.0,
    v_max: float = 4500.0,
) -> float:
    """
    Evaluate a single latent vector: decode → forward sim → reward.
    Returns scalar reward (higher is better).
    """
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
    v_model = vae_decoder.decode(z_t).squeeze(0).squeeze(0)  # [nx, nz]

    # Prior reward
    dx = torch.diff(v_model, dim=0)
    dz = torch.diff(v_model, dim=1)
    smooth = -(dx.pow(2).mean() + dz.pow(2).mean())
    bound = -torch.clamp(v_min - v_model, min=0).pow(2).mean() - torch.clamp(v_model - v_max, min=0).pow(2).mean()
    r_prior = smooth + bound

    # Forward simulation
    from agents.transmission_forward import AcquisitionForward
    geom = AcquisitionGeometry(
        nx_model=v_model.shape[0], nz_model=v_model.shape[1],
        dx=10.0, dt=0.001, freq=15.0, nt=1000, n_shots=5, n_receivers=70,
        pml_width=40, geometry="reflection",
    )
    forward = AcquisitionForward(geom)
    p_pred = forward.simulate(v_model, device=device)  # [nt, n_recv]

    # FWI reward (add batch dim for reward functions)
    p_pred_b = p_pred.unsqueeze(0)  # [1, nt, n_recv]
    p_obs_b = p_obs.unsqueeze(0)
    r_l1 = reward_l1(p_pred_b, p_obs_b).item()
    r_l2 = reward_l2(p_pred_b, p_obs_b).item()

    total_reward = r_l1 + r_l2 + prior_weight * r_prior.item()
    return total_reward


def evaluate_population(
    Z: np.ndarray,
    vae_decoder,
    p_obs: torch.Tensor,
    v_true: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, float]:
    """Evaluate a population of latent vectors. Returns (rewards, best_mae)."""
    rewards = np.zeros(len(Z))
    best_mae = float("inf")
    for i, z in enumerate(Z):
        r = evaluate_z(z, vae_decoder, p_obs, device)
        rewards[i] = r
        # Also compute MAE for logging
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
        v_model = vae_decoder.decode(z_t).squeeze(0).squeeze(0)
        mae = (v_model - v_true).abs().mean().item()
        if mae < best_mae:
            best_mae = mae
    return rewards, best_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--model_source", choices=["cva", "fva"], default="cva")
    parser.add_argument("--cva_file_idx", type=int, default=50)
    parser.add_argument("--cva_sample_idx", type=int, default=0)
    parser.add_argument("--cva_root", type=str, default="data/CVA/CurveVel_A")
    parser.add_argument("--fva_root", type=str, default="data/FVA_model")
    parser.add_argument("--max_generations", type=int, default=200)
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--init_sigma", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="runs/cmaes")
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # Load VAE decoder
    print(f"Loading VAE: {args.vae_ckpt}")
    vae_decoder = VAEDecoder(args.vae_ckpt, device=device)
    latent_dim = vae_decoder.latent_dim
    print(f"  latent_dim={latent_dim}")

    # Load velocity model and generate observation
    if args.model_source == "cva":
        from utils.data_loader import CurveVelADataset
        cva = CurveVelADataset(args.cva_root)
        item = cva.get_by_file_index(args.cva_file_idx, args.cva_sample_idx)
        v_true_np = item["model"].astype(np.float32)
    else:
        import glob, re
        fva_files = sorted(glob.glob(os.path.join(args.fva_root, "model*.npy")),
                           key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
        fva_data = np.load(fva_files[args.cva_file_idx], mmap_mode="r")
        v_true_np = np.asarray(fva_data[args.cva_sample_idx], dtype=np.float32).copy()
        if v_true_np.ndim == 3:
            v_true_np = v_true_np[0]

    geom = AcquisitionGeometry(
        nx_model=v_true_np.shape[0], nz_model=v_true_np.shape[1],
        dx=10.0, dt=0.001, freq=15.0, nt=1000, n_shots=5, n_receivers=70,
        pml_width=40, geometry="reflection",
    )
    p_obs, v_true = make_observation(v_true_np, geom, device)
    print(f"Model: {args.model_source}[{args.cva_file_idx}], v_range=[{v_true_np.min():.0f}, {v_true_np.max():.0f}]")

    # Initialize CMA-ES
    x0 = np.zeros(latent_dim)
    es = cma.CMAEvolutionStrategy(
        x0, args.init_sigma,
        {
            "popsize": args.pop_size,
            "maxiter": args.max_generations,
            "verbose": -1,  # quiet
            "seed": 42,
        },
    )

    best_reward = -float("inf")
    best_z = x0.copy()
    best_mae = float("inf")
    best_gen = 0
    history = []

    print(f"\n{'='*60}")
    print(f"CMA-ES: pop={args.pop_size}, max_gen={args.max_generations}, sigma0={args.init_sigma}")
    print(f"{'='*60}")

    t_start = time.time()
    gen = 0
    while not es.stop():
        Z = es.ask()  # population of candidates
        rewards, gen_best_mae = evaluate_population(Z, vae_decoder, p_obs, v_true, device)
        es.tell(Z, [-r for r in rewards])  # CMA-ES minimizes, so negate

        gen_best_idx = np.argmax(rewards)
        gen_reward = rewards[gen_best_idx]

        if gen_best_mae < best_mae:
            best_mae = gen_best_mae
            best_z = Z[gen_best_idx].copy()
            best_reward = gen_reward
            best_gen = gen

        elapsed = time.time() - t_start
        history.append({"gen": gen, "best_mae": best_mae, "best_reward": best_reward,
                        "gen_mae": gen_best_mae, "sigma": es.sigma})

        if gen % 10 == 0 or gen < 5:
            print(f"  gen {gen:4d} | best_MAE={best_mae:.1f} @gen{best_gen} | "
                  f"gen_MAE={gen_best_mae:.1f} | σ={es.sigma:.3f} | "
                  f"rewards mean={rewards.mean():.0f} | {elapsed:.0f}s")

        gen += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"CMA-ES complete: {gen} generations in {elapsed:.0f}s")
    print(f"Best MAE: {best_mae:.1f} @ generation {best_gen}")
    print(f"{'='*60}")

    # Save best result
    best_v = vae_decoder.decode(
        torch.from_numpy(best_z.astype(np.float32)).unsqueeze(0).to(device)
    ).squeeze().cpu().numpy()
    np.save(os.path.join(args.out_dir, "best_velocity.npy"), best_v)
    np.save(os.path.join(args.out_dir, "best_z.npy"), best_z)

    # Save history
    hist_arr = np.array([(h["gen"], h["best_mae"], h["sigma"]) for h in history])
    np.savetxt(os.path.join(args.out_dir, "history.csv"), hist_arr,
               delimiter=",", header="gen,best_mae,sigma", comments="")

    print(f"Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
