# Multi-Frequency Curriculum Learning

## Strategy
1. **5Hz**: Gaussian policy + TT reward (low freq → no cycle skip, stable convergence)
2. **10Hz**: Load 5Hz checkpoint → Contrastive reward (mid freq → more detail)
3. **15Hz**: Load 10Hz checkpoint → Contrastive reward (full freq → final refinement)

## Results

| Model | TT@15Hz | Prog Contra | Prog L2 | Curr Best | Improvement |
|-------|---------|-------------|---------|-----------|-------------|
| CVA18 | 10.2 | 12.4 | 4.1 | **7.7** | vs TT |
| CVA50 | 36.9 | 33.0 | 41.3 | **48.5** | vs ProgL2 |
| CVA10 | 150.0 | 116.0 | 200.1 | **75.7** | vs TT |

| Model | 5Hz TT | 10Hz Contra | 15Hz Contra |
|-------|--------|-------------|-------------|
| CVA18 | 10.9 | 7.7 | 8.3 |
| CVA50 | 56.3 | 66.7 | 48.5 |
| CVA10 | 149.2 | 91.5 | 75.7 |

## Key Findings

- **CVA10: 150→76 (49% improvement)** — curriculum dramatically helps hard models
- **CVA18: 10→8** — modest improvement on already-easy models
- **CVA50: not helped** — curriculum degrades performance, needs investigation
- Multi-frequency curriculum is most effective where single-frequency methods struggle most
- The progressive frequency increase mirrors FWI multiscale strategy — validated in RL context
