# Band-Filtered Curriculum

## Method
Same 15Hz p_data throughout. Reward computed on progressively wider frequency bands:
1. **0-5Hz**: TT reward (low-freq only, stable convergence)
2. **0-10Hz**: Contrastive reward (adds mid-freq detail)
3. **Full**: Contrastive reward (all frequencies)

## Results

| Model | 0-5Hz TT | 0-10Hz Contra | Full Contra | TT@15Hz | ProgContra |
|-------|----------|---------------|-------------|---------|------------|
| CVA18 | 10.2 | 11.7 | 13.8 | 10.2 | 12.4 |
| CVA50 | 36.9 | 49.9 | 45.5 | 36.9 | 33.0 |
| CVA10 | 150.0 | 196.1 | 89.0 | 150.0 | 116.0 |

## Key Findings

- **CVA10: 150→89 (41% improvement)** — band curriculum beats single-stage ProgContra (116)
- **CVA18/50: no improvement** — lowpass filtering blurs already-sufficient TT signal
- Band filtering is less effective than multi-frequency forward simulation
- But proves the concept: frequency-band reward shaping helps on cycle-skip-prone models
