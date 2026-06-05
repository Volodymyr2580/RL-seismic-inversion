# CHANGELOG

## 2026-05-14 — Phase II launch: Beta policy + GDPO-Guard + transmission FWI

### Repository reorganization
- Moved all Phase I scripts to `archive/old_scripts/`:
  - `train_single_observation_rl.py`, `train_transmission_rl.py`, `pretrain_cva_continuous_policy.py`
  - `run_fwi_ablation_local.py`, `run_param_sweep.py`, `compare_reward_scales.py`
  - `run_sweep.sh`, `run_sweep_discrete.sh`, `run_sweep_si.sh`, `run_sweep_transmission.sh`
- Moved `single_observation_rl_experiments.md`, `rl_policy_gradient_grpo_family_talk.md` → `archive/`
- Deleted old `memory-bank/progress.md`, created fresh Phase II version

### New Phase II code
- `agents/cnn_encoder.py`: Lightweight CNN (~50K params) for seismic → control grid features
- `agents/beta_policy.py`: Beta(α,β) distribution policy (CNN-conditioned + learnable variants)
- `agents/transmission_forward.py`: Transmission-geometry forward simulator (sources top, receivers bottom, not in PML)
- `train_rl_fwi.py`: Phase II main training entry (Beta + GDPO-Guard + L1/L2 split rewards)
- `pretrain_cnn_cva.py`: CVA CNN encoder pretraining (MSE regression, seismic → B-spline control points)

### Modified files
- `agents/rl_objectives.py`: Complete rewrite — GDPO advantage, GRPO-Guard ratio correction, L1/L2 split rewards, Beta-compatible log-ratio
- `plan.md`: Complete rewrite — Phase II focus (CNN+Beta+GDPO-Guard+multi-reward)
- `AGENTS.md`: Simplified to Phase II pipeline

### Key design decisions
- **Beta distribution** over Gaussian+sigmoid: natural bounded support [0,1], no gradient vanishing at boundaries
- **L1/L2 as separate GDPO components**: per-reward group normalization preserves finer advantage granularity
- **GRPO-Guard ratio correction**: centering + EMA monitoring to combat implicit over-optimization
- **Transmission geometry**: sources at surface (z=0), receivers at bottom (z=nz-1), validated outside PML
- **Phase II rewards**: FWI-L1 + FWI-L2 only; SI, prior, well-log deferred to Phase III

### Known limitations
- deepwave not available in current env → smoke test needs GPU server
- deepwave does not support velocity-model batching → G models simulated serially
- Beta distribution log_prob can be numerically unstable near 0/1 → clamped with 1e-7 margin

## 2026-05-07 — Repository reorganization

### Cleaned up
- Old scripts (train.py, train_v1.py, train_DAPO.py, toy_train.py, toy_train_DAPO.py, test.py) → `archive/old_scripts/`
- Old shell scripts (run_reward_ablation.sh, run_si_only_explore.sh) → `archive/old_scripts/`
- Jupyter notebooks → `notebooks/`
- Reference PDF → `papers/`
- Old experiment results (RL_results0, RL_results_fix, results, grpo_dapo_v1) → `archive/old_results/`
- Old runs (all smoke/fix/toy runs) → `archive/old_runs/`
- `RL_results_v2_ppofix/` → renamed to `results/`

### Root now contains
- 4 active Python scripts: train_single_observation_rl.py, pretrain_cva_continuous_policy.py, compare_reward_scales.py, run_fwi_ablation_local.py
- 5 docs: AGENTS.md, plan.md, CHANGELOG.md, single_observation_rl_experiments.md
- Active dirs: agents/, config/, data/, utils/, example_to_dyy/, memory-bank/, outputs/, results/, runs/

## 2026-05-07 — Critical Bug Fix: PPO/GRPO old_policy sync timing

### Bug
`train_single_observation_rl.py` synced `old_policy` to `policy` at the **start** of each step, then used `old_policy` for rollout and `policy` for `logp_new`. Because only ONE gradient update was performed per step, `logp_old == logp_new` and the importance ratio was always 1.0, making the clipped surrogate loss collapse to `-mean(advantages) ≈ 0`. All RL_results0 experiments (continuous/discrete, FWI/SI, all annealing schedules) were affected — no meaningful policy learning occurred.

### Fix
1. **Restructured the training loop into 4 phases** (see `train_single_observation_rl.py` ~L610-710):
   - Phase 1: Rollout with frozen `old_policy` snapshot.
   - Phase 2: K PPO epochs on the same batch. Epoch 0 has ratio=1 (wasted), epochs 1+ have meaningful ratio ≠ 1.
   - Phase 3: Post-update diagnostics (entropy, best model, MAE/RMSE).
   - Phase 4: Sync `old_policy` to updated policy for the next rollout batch.
2. **Added `--ppo_epochs` argument** (default: 4).
3. **Entropy computation moved** to post-update (Phase 3) rather than pre-update.

### Files changed
- `train_single_observation_rl.py`: main training loop rewritten; `--ppo_epochs` added.

### Verification
- AST syntax check passed.
- User should run Step 1 verification: `--steps 50 --ppo_epochs 4 --log_every 1` and check that `ratio_mean` deviates from 1.0 and `clip_fraction > 0`.
