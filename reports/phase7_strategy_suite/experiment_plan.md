# Phase7 Strategy Suite Plan

Purpose: test why high-dimensional B-spline RL degraded good initial models in the Phase7 reward suite.

Core questions:

1. Does the degradation mainly come from excessive exploration?
2. Does the high-dimensional joint PPO ratio become unstable when control points increase from 16 to 300/432?
3. Does an initialization tether help turn `near_blur` into local refinement instead of destructive exploration?
4. Is a smaller B-spline control grid more stable than the full `30x10` / `36x12` grid?

Models:

- `marmousi`: `nx=210,nz=70,ctrl_full=30x10`
- `salt_A_90x270`: `nx=270,nz=90,ctrl_full=36x12`
- `synthetic`: `nx=210,nz=70,ctrl_full=30x10`

Initial models:

- `far_uniform`
- `medium_gradient`
- `near_blur`

Rewards:

- `tt_only`
- `ncc_zero`
- `wasserstein_w2`

Strategies:

- `full_low_joint`: full control grid, joint action PPO ratio, low exploration.
- `full_token_high`: full control grid, per-control-point PPO ratio, original high exploration.
- `full_token_low`: full control grid, token ratio, low exploration.
- `full_token_tether`: full grid, token ratio, low exploration, weak init tether.
- `full_token_tether_strong`: full grid, token ratio, very low exploration, strong init tether.
- `coarse_token_low`: coarse grid, token ratio, low exploration.
- `mid_token_low`: mid grid, token ratio, low exploration.

Default run size:

- Full-grid strategies: `3 models x 3 inits x 3 rewards x 5 strategies = 135`
- Grid-scale strategies: `3 models x 3 inits x 2 rewards x 2 strategies = 36`
- Total: `171` runs

Primary metric:

```text
delta_best_vs_init = best_mae_global - initial_model_mae
```

Interpretation:

- Negative delta: training improved over the deterministic initial model.
- Positive delta: training degraded the deterministic initial model.

Expected report:

- `reports/phase7_strategy_suite/index.html`
- `reports/phase7_strategy_suite/summary.csv`
- `reports/phase7_strategy_suite/summary.json`
- Delta heatmaps per model/init.
