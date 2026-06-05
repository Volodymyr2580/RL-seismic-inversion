"""
Diagnose ratio≈1.0 issue in latent RL.

Tests:
1. Policy parameter change between steps
2. logp_old vs logp_new difference
3. Advantage distribution
4. Loss components (surrogate loss, clipping effect)
5. Gradient flow
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from agents.latent_policy import LearnableLatentPolicy, VAEDecoder
from agents.rl_objectives import gdpo_advantage, clipped_policy_loss, RewardWeights


def test_parameter_change():
    """Test: does the policy actually change after a PPO update?"""
    print("=" * 60)
    print("TEST 1: Parameter change after PPO update")
    print("=" * 60)

    policy = LearnableLatentPolicy(latent_dim=64, init_sigma=0.5)
    params_before = {n: p.clone() for n, p in policy.named_parameters()}

    # Simulate: sample, compute fake advantage, PPO update
    G = 8
    with torch.no_grad():
        z = policy.sample(G)
        logp_old = policy.log_prob(z)  # [G, 64]

    # Fake advantage: all positive large
    adv = torch.ones(G, 1) * 5.0

    # PPO update
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    for epoch_i in range(2):
        policy.train()
        logp_new = policy.log_prob(z)
        logp_new_4d = logp_new.unsqueeze(1).unsqueeze(-1)  # [G, 1, 64, 1]
        logp_old_4d = logp_old.unsqueeze(1).unsqueeze(-1)
        loss, stats = clipped_policy_loss(
            logp_new=logp_new_4d,
            logp_old=logp_old_4d,
            advantages=adv,
            epsilon_low=0.20,
            epsilon_high=0.27,
            token_mean=False,
        )
        print(f"  Epoch {epoch_i}: loss={loss.item():.6f}, ratio_mean={stats['ratio_mean']:.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    params_after = {n: p.clone() for n, p in policy.named_parameters()}
    for n in params_before:
        diff = (params_after[n] - params_before[n]).abs().mean().item()
        print(f"  Δ{n}: {diff:.8f}")

    # Check logp change
    with torch.no_grad():
        logp_after = policy.log_prob(z)
    diff_lp = (logp_after - logp_old).abs().mean().item()
    print(f"  |Δlogp| mean: {diff_lp:.8f}")
    ratio_per_dim = torch.exp(logp_after - logp_old)
    joint_ratio = torch.exp((logp_after - logp_old).sum(dim=1))
    print(f"  per-dim ratio mean: {ratio_per_dim.mean():.6f}")
    print(f"  joint ratio mean: {joint_ratio.mean():.6f}")
    print()


def test_advantage_scale():
    """Test: how does GDPO scale FWI reward?"""
    print("=" * 60)
    print("TEST 2: GDPO advantage scaling")
    print("=" * 60)

    # Simulate realistic FWI rewards
    G = 8
    torch.manual_seed(42)
    r_l1 = torch.randn(G) * 10000 - 1.7e6  # realistic R_L1 range
    r_l2 = torch.randn(G) * 100000 - 2.2e7  # realistic R_L2 range
    r_prior = torch.randn(G) * 100 - 800

    print(f"  R_L1: mean={r_l1.mean():.0f}, std={r_l1.std():.0f}, range=[{r_l1.min():.0f}, {r_l1.max():.0f}]")
    print(f"  R_L2: mean={r_l2.mean():.0f}, std={r_l2.std():.0f}")
    print(f"  R_prior: mean={r_prior.mean():.1f}, std={r_prior.std():.1f}")

    weights = RewardWeights(l1=1.0, l2=1.0, si=0.0, prior=0.05)
    adv = gdpo_advantage(
        {"l1": r_l1.unsqueeze(1), "l2": r_l2.unsqueeze(1),
         "si": torch.zeros(G, 1), "prior": r_prior.unsqueeze(1)},
        weights,
        batch_norm=False,
    )
    print(f"  GDPO advantage: mean={adv.mean():.4f}, std={adv.std():.4f}, range=[{adv.min():.4f}, {adv.max():.4f}]")
    print()


def test_realistic_ratio():
    """Test: with realistic advantage scale, what ratio/clip do we get?"""
    print("=" * 60)
    print("TEST 3: Realistic PPO with GDPO advantage")
    print("=" * 60)

    policy = LearnableLatentPolicy(latent_dim=64, init_sigma=0.5)
    G = 8

    with torch.no_grad():
        z = policy.sample(G)
        logp_old = policy.log_prob(z)

    # Realistic GDPO advantage
    torch.manual_seed(42)
    r_l1 = torch.randn(G) * 10000 - 1.7e6
    r_l2 = torch.randn(G) * 100000 - 2.2e7
    r_prior = torch.randn(G) * 100 - 800
    weights = RewardWeights(l1=1.0, l2=1.0, si=0.0, prior=0.05)
    adv = gdpo_advantage(
        {"l1": r_l1.unsqueeze(1), "l2": r_l2.unsqueeze(1),
         "si": torch.zeros(G, 1), "prior": r_prior.unsqueeze(1)},
        weights, batch_norm=False,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    for ppo_i in range(4):
        logp_new = policy.log_prob(z)
        logp_new_4d = logp_new.unsqueeze(1).unsqueeze(-1)
        logp_old_4d = logp_old.unsqueeze(1).unsqueeze(-1)

        # Per-dimension log_ratio stats (before clipping)
        with torch.no_grad():
            per_dim_ratio = torch.exp(logp_new - logp_old)
            print(f"  Epoch {ppo_i}: per-dim ratio mean={per_dim_ratio.mean():.4f}, "
                  f"std={per_dim_ratio.std():.4f}, range=[{per_dim_ratio.min():.4f}, {per_dim_ratio.max():.4f}]")

        loss, stats = clipped_policy_loss(
            logp_new=logp_new_4d, logp_old=logp_old_4d,
            advantages=adv, epsilon_low=0.20, epsilon_high=0.27,
            token_mean=False,
        )
        print(f"    loss={loss.item():.6f}, ratio_mean={stats['ratio_mean']:.6f}, "
              f"clip={stats['clip_fraction']:.6f}")

        optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
        print(f"    grad_norm={grad_norm:.6f}")
        optimizer.step()
    print()


def test_entropy_effect():
    """Test: does entropy bonus prevent updates?"""
    print("=" * 60)
    print("TEST 4: Entropy bonus effect")
    print("=" * 60)

    policy = LearnableLatentPolicy(latent_dim=4, init_sigma=0.5)  # small for clarity
    G = 4
    with torch.no_grad():
        z = policy.sample(G)
        logp_old = policy.log_prob(z)

    adv = torch.tensor([[2.0], [1.0], [-1.0], [-2.0]])

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    for entropy_bonus in [0.0, 0.02, 0.1]:
        # Reset policy
        policy = LearnableLatentPolicy(latent_dim=4, init_sigma=0.5)
        with torch.no_grad():
            z = policy.sample(G)
            logp_old = policy.log_prob(z)

        for epoch_i in range(2):
            logp_new = policy.log_prob(z)
            logp_new_4d = logp_new.unsqueeze(1).unsqueeze(-1)
            logp_old_4d = logp_old.unsqueeze(1).unsqueeze(-1)

            loss, stats = clipped_policy_loss(
                logp_new=logp_new_4d, logp_old=logp_old_4d,
                advantages=adv, epsilon_low=0.20, epsilon_high=0.27,
                token_mean=False,
            )
            ent_tensor = policy.entropy().mean()
            loss_total = loss - entropy_bonus * ent_tensor

            optimizer_2 = torch.optim.AdamW(policy.parameters(), lr=1e-2)
            optimizer_2.zero_grad()
            loss_total.backward()
            optimizer_2.step()

        # Final ratio
        with torch.no_grad():
            logp_final = policy.log_prob(z)
            joint_ratio = torch.exp((logp_final - logp_old).sum(dim=1))
            per_dim_ratio = torch.exp(logp_final - logp_old)
        print(f"  entropy_bonus={entropy_bonus:.2f}: joint_ratio={joint_ratio.mean():.4f}, "
              f"per_dim_ratio_mean={per_dim_ratio.mean():.4f}, "
              f"entropy={policy.entropy().mean():.4f}")
    print()


if __name__ == "__main__":
    test_parameter_change()
    test_advantage_scale()
    test_realistic_ratio()
    test_entropy_effect()
