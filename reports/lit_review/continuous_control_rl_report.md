# Continuous Control RL and Our Framework: Context and Relationship

## 1. What the Reviewer Meant

The reviewer flagged that our Introduction traced an RL lineage from `Policy Gradient → TRPO → PPO → GRPO/GDPO`, but **GRPO and GDPO were designed for LLM alignment** (discrete token-level RL, e.g., DeepSeekMath, SimPO), not for continuous control (robotics, game playing, PDE optimization). The reviewer argued this lineage is historically inaccurate and that we should instead cite the continuous-control RL literature: **SAC (Soft Actor-Critic), TD3 (Twin Delayed DDPG), DDPG (Deep Deterministic Policy Gradient)** — algorithms that, like our work, optimize continuous action spaces.

## 2. The Actual Relationship

### 2.1 Our Setup is Continuous Control

Our policy outputs **continuous velocity values** (control points in ℝ), sampled from a Gaussian distribution. This is textbook continuous control:
- **State**: Observation seismogram $p_{\text{obs}}$ (fixed — single-sample inversion)
- **Action**: Velocity model parameters $c \in \mathbb{R}^{16}$ or $z \in \mathbb{R}^{64}$
- **Environment**: Deepwave forward simulator (black box)
- **Reward**: Physics-based misfit (TT, L2, Contrastive, etc.)

The action space is **continuous and multi-dimensional** — exactly the domain where SAC, TD3, and PPO were developed.

### 2.2 Why We Use PPO, Not SAC/TD3

| Algorithm | Requires | Our Case |
|-----------|----------|----------|
| SAC | Learned Q-function (critic) + replay buffer | We have no replay buffer — each rollout is expensive (deepwave forward) |
| TD3 | Twin critics + target networks + replay buffer | Same issue — sample-inefficient for expensive environments |
| PPO | Only the policy + old log-probs | On-policy, no replay buffer needed. Works well with few samples. |

PPO is the natural choice for **expensive, on-policy environments** where each forward simulation costs seconds and we can only afford G=32 parallel rollouts. SAC/TD3 would need orders of magnitude more environment interactions.

### 2.3 What We Borrow from GDPO (Methods Section, not Introduction)

GDPO's contribution is a **multi-reward advantage decomposition**:

$$A_{\text{total}} = \sum_i w_i \cdot \frac{R_i - \mu_i}{\sigma_i}$$

where each reward component is independently group-standardized before being summed. This prevents a single large-magnitude component (e.g., L2 at ~$10^7$) from drowning out a small-magnitude component (e.g., TT at ~$0.1$).

We do NOT use GDPO's LLM-specific features (token-level ratios, preference optimization, reference policy). We only borrow this **multi-reward normalization trick**, which is problem-agnostic and applies to any domain where multiple reward components of different scales must be balanced.

## 3. What Should Go in the Introduction

The RL lineage in the Introduction should be:

1. **Policy Gradient Theorem** [Sutton et al., 1999] — foundation
2. **TRPO** [Schulman et al., 2015] — trust region for stable updates
3. **PPO** [Schulman et al., 2017] — clipped surrogate objective, the standard for continuous control
4. Briefly mention: PPO is chosen because (a) it works with on-policy data from expensive simulators, (b) no replay buffer or value function critic is required, (c) the clipped objective prevents destructive policy updates.

Do NOT mention GRPO/GDPO in the Introduction. They belong in the **Methods section**, where we explain that we adopt GDPO's multi-reward group-normalization technique to balance heterogeneous reward components, without using its LLM-specific machinery.

## 4. What Should Go in the Methods Section

```
We adopt a multi-reward advantage computation inspired by Group Direct 
Policy Optimization (GDPO) [ref]. Each reward component R_k (e.g., 
travel-time misfit, waveform L2 norm) is independently standardized 
across the group of G candidate models:

    A_k = (R_k - μ_k) / σ_k

The total advantage is a weighted sum: A = Σ w_k · A_k. This 
decoupled normalization prevents reward components with large 
numerical magnitudes from dominating the optimization, and is 
applied here as a domain-agnostic technique for multi-objective 
RL — independent of GDPO's original LLM-alignment context.
```

## 5. Should We Cite SAC/TD3/DDPG?

**Yes, briefly in the Introduction**, to show awareness of the continuous-control literature and to justify our choice of PPO. One paragraph:

```
The continuous-control RL literature offers several alternatives, 
including Deep Deterministic Policy Gradient (DDPG) [ref], Twin 
Delayed DDPG (TD3) [ref], and Soft Actor-Critic (SAC) [ref]. These 
algorithms achieve state-of-the-art sample efficiency on benchmark 
continuous control tasks but rely on experience replay buffers and 
learned Q-function critics that require orders of magnitude more 
environment interactions than are feasible with expensive seismic 
forward simulations. Proximal Policy Optimization (PPO) [ref], by 
contrast, is an on-policy method that makes efficient use of limited 
rollout data and requires no critic network, making it the natural 
choice for computationally expensive PDE-constrained environments.
```

## 6. Summary

| Element | Where | Why |
|---------|-------|-----|
| Policy Gradient → TRPO → PPO | Introduction | Standard RL lineage |
| SAC/TD3/DDPG + why not used | Introduction | Shows awareness, justifies PPO choice |
| GDPO multi-reward normalization | Methods | The technique we actually use |
| GRPO/GDPO in RL lineage | **Nowhere** | They are LLM-alignment algorithms, not our domain |

The core message: our contribution is the **RL-for-FWI framework**, not a new RL algorithm. We use standard PPO with a borrowed multi-reward normalization technique. The novelty is in applying this to seismic inversion, not in advancing RL theory.
