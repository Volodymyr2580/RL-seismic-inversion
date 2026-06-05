# Progressive TT → Contrastive

## Strategy
1. Train Gaussian policy with TT reward (log-scaled energy-ratio picker)
2. Load best TT checkpoint
3. Continue training with Contrastive reward (spectrum similarity + cross-correlation)
Lower initial temperature (1.0→0.1, 500 steps) since policy is already near optimum.

## Results

| Model | Range | TT-only | Prog TT→L2 | Prog TT→Contra | vs TT | vs ProgL2 |
|-------|-------|---------|------------|----------------|-------|----------|
| CVA18 | [1666,2848] | 10.2 | 4.1 | 12.4 | +2.2 | +8.3 |
| CVA50 | [1987,2407] | 36.9 | 41.3 | 33.0 | -3.9 | -8.2 |
| CVA10 | [1895,3726] | 150.0 | 200.1 | 116.0 | -34.0 | -84.1 |
| CVA6 | [1737,3510] | 58.4 | 51.7 | 54.8 | -3.6 | +3.1 |
| CVA8 | [2826,3475] | 75.1 | 69.1 | 94.8 | +19.7 | +25.7 |
| CVA5 | [1705,3446] | 106.0 | 141.2 | 80.9 | -25.1 | -60.3 |

## Key Findings

- Beat TT-only on 4/6 models
- Beat Prog L2 on 3/6 models
- **Strong on hard models**: CVA10 (150→116, -23%), CVA5 (106→81, -24%)
- **Weak on easy models**: CVA18 (10→12, worse) — Contrastive lacks L2s precision for already-good models
- **Recommendation**: Use Prog L2 for well-constrained models, Prog Contra for difficult ones
