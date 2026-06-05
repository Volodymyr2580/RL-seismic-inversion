# Phase IV Final Report — Complete

## Results (9 strategies × 6 models)

| Strategy | CVA18 | CVA50 | CVA10 | CVA6 | CVA8 | CVA5 |
|---|---|---|---|---|---|---|
| TT-only | 10.2 | 36.9 | 150.0 | 58.4 | 75.1 | 106.0 |
| Prog_TT_to_L2 | 4.1 | 41.3 | 200.1 | 51.7 | 69.1 | 141.2 |
| Multi_TT+L2 | 45.8 | 31.9 | 159.7 | 42.4 | 69.8 | 139.7 |
| L1+L2 | 83.1 | 39.2 | 247.3 | 33.5 | 22.8 | 212.6 |
| FWI_L2 | 339.2 | 188.3 | 259.8 | 270.6 | 40.2 | 273.9 |
| FWI_Envelope | 60.8 | 41.9 | 163.6 | 38.3 | 84.9 | 148.7 |
| FWI_Windowed | 372.4 | 325.4 | 234.2 | 270.8 | 104.0 | 278.5 |
| FWI_Wasserstein | 365.1 | 241.1 | 228.1 | 108.4 | 115.8 | 277.2 |
| FWI_Contrastive | 38.3 | 27.6 | 97.5 | 65.8 | 130.3 | 90.3 |

## Best per Model

- **CVA18**: 4.1 (Prog_TT_to_L2)
- **CVA50**: 27.6 (FWI_Contrastive)
- **CVA10**: 97.5 (FWI_Contrastive)
- **CVA6**: 33.5 (L1+L2)
- **CVA8**: 22.8 (L1+L2)
- **CVA5**: 90.3 (FWI_Contrastive)

## Key Findings

1. **Progressive TT→L2 is best for easy models** (CVA18: 4.1)
2. **Contrastive reward excels on hard models** (CVA10: 97.5, CVA5: 90.3)
3. **L1+L2 is competitive** (CVA8: 22.8, CVA6: 33.5)
4. **No single strategy dominates** — best approach depends on model complexity
5. **Envelope/Wasserstein/Windowed fail from random init** — need good starting point
6. **TT-only is the most robust baseline** — never catastrophic
