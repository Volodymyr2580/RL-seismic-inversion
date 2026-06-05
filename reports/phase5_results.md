# Phase 5A Results Report

**日期**: 2026-05-26  
**实验**: 5 个新 reward × 6 个 CVA 模型，5000 步，G=32  

---

## 1. 完整结果表

| Strategy | CVA18 | CVA50 | CVA10 | CVA6 | CVA8 | CVA5 | **Mean** |
|----------|-------|-------|-------|------|------|------|----------|
| **Wasserstein_abs** | **4.5** | 51.2 | **146.3** | 41.6 | 47.6 | 110.4 | **67.0** |
| NCC_maxlag | 21.3 | 57.0 | 147.4 | 58.0 | 74.3 | 114.5 | 78.7 |
| AWI | 22.8 | **50.4** | 147.1 | 53.9 | 75.6 | 117.5 | 77.9 |
| Envelope_NCC | 23.2 | 62.1 | 148.5 | 48.9 | 75.9 | 123.7 | 80.4 |
| NCC_zero | 37.2 | 244.6 | 186.7 | **31.6** | 78.0 | 114.0 | 115.3 |

**Phase IV 基线（对比）:**

| Strategy | CVA18 | CVA50 | CVA10 | CVA6 | CVA8 | CVA5 | **Mean** |
|----------|-------|-------|-------|------|------|------|----------|
| TT-only | 10.2 | 36.9 | 150.0 | 58.4 | 75.1 | 106.0 | 72.8 |
| Contrastive | 38.3 | **27.6** | **97.5** | 65.8 | 130.3 | **90.3** | 75.0 |
| Prog_TT→L2 | **4.1** | 41.3 | 200.1 | 51.7 | 69.1 | 141.2 | 84.6 |
| L1+L2 | 83.1 | 39.2 | 247.3 | 33.5 | **22.8** | 212.6 | 106.4 |
| Wass_BUG(old) | 365.1 | 241.1 | 228.1 | 108.4 | 115.8 | 277.2 | 222.6 |

---

## 2. 核心发现

### 发现 1: Wasserstein_abs 是 Phase 5A 的明显赢家

- **Mean = 67.0** — 所有策略（含 Phase IV）中最低的均值，击败 TT-only（72.8）
- **CVA18 = 4.5** — 几乎追平 Prog_TT→L2 的 4.1，但 Wasserstein 是**单一 reward**，不需要 curriculum
- 这是最"鲁棒"的单一策略：6 个模型上没有一个崩盘（最差 CVA10=146.3）

### 发现 2: Bug 修复效果巨大

- Wass_BUG(old) mean = 222.6 → Wass_fixed mean = 67.0
- **修复一个 bug 带来 3.3 倍提升**，从"完全不能用"变成"综合最优"
- 证明 CDF-based time-domain W₁ 在 FWI 中是有效的鲁棒 objective

### 发现 3: NCC_maxlag 和 AWI 中等有效

- NCC_maxlag mean=78.7, AWI mean=77.9 — 与 Contrastive (75.0) 相当
- 但都不及 Wasserstein_abs 鲁棒

### 发现 4: NCC_zero 是最差的

- Mean=115.3 — 零时移 NCC 单独用不够，cycle-skipping 时 ncc₀ 接近 0
- 证明了 nonzero-lag 搜索（NCC_maxlag）的必要性

### 发现 5: 单一 reward 无法通吃所有模型

- Phase IV 的 curriculum 策略（Prog_TT→L2）在简单模型上仍然最好（4.1）
- Phase IV 的 Contrastive 在困难模型上仍然最好（CVA50=27.6, CVA10=97.5, CVA5=90.3）
- 但 Wasserstein_abs 是"最平衡"的选择——没有任何模型表现灾难性

---

## 3. 每个模型的 Best

| Model | Best MAE | Strategy | Phase |
|-------|----------|----------|-------|
| CVA18 | **4.1** | Prog_TT→L2 | IV |
| CVA50 | **27.6** | Contrastive | IV |
| CVA10 | **97.5** | Contrastive | IV |
| CVA6 | **31.6** | NCC_zero | **5A** |
| CVA8 | **22.8** | L1+L2 | IV |
| CVA5 | **90.3** | Contrastive | IV |

Phase 5A 仅在 CVA6 上超过 Phase IV 最佳。

---

## 4. Phase 5B 建议

基于 Phase 5A 结果，curriculum 组合方向：

1. **Wasserstein → Contrastive**：先用 W₁ 抓背景速度（鲁棒、不崩），再用 Contrastive 细调（Phase IV 在难模型上最好）
2. **Wasserstein → L1+L2**：简单模型上可能效果更好（CVA8 上 L1+L2=22.8）
3. **Wasserstein + TT + Contrastive**：三阶段递进

Wasserstein_abs 的鲁棒性使其成为 curriculum 的第一步的理想选择——它永远不会崩，给后续策略提供一个合理的初始点。

---

## 5. 论文叙事价值

Phase 5 实验为论文贡献了清晰的 story arc：

1. **L2 waveform misfit 的问题**：Phase IV 显示 L2-only mean=228，cycle-skipping 严重
2. **文献指导的改进方向**：导师指出的 cross-correlation / time-lag / OT 方向
3. **OT (Wasserstein) 的潜力与陷阱**：错误实现 → 完全失败 (mean=222.6)；正确实现 → 综合最优 (mean=67.0)
4. **RL 框架的优势**：CDF-based W₁ 不需要 adjoint source，在 RL reward 中直接可用
5. **单 reward 的局限**：curriculum 在极端模型上仍有优势，指出未来方向

---

## 6. 待生成可视化

- [ ] Wasserstein_abs 6 模型 progression 图
- [ ] Phase IV vs Phase 5A 对比图
- [ ] Bug fix before/after 对比图
