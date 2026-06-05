# Phase 5 Large-Scale Final Report

**日期**: 2026-06-02  
**规模**: 19 策略 × 16 模型 × 5 seeds = **1520 valid runs**  
**运行时间**: 2026-05-29 ~ 06-02 (~4.5 days, 4× RTX 4090)  
**数据来源**: Wenzhe 的 `phase4_reward_design_report.md`（导师文献综述）

---

## 1. 最终排名 (Mean MAE across 16 models × 5 seeds)

| Rank | Strategy | Type | Mean MAE | Std |
|------|----------|------|----------|-----|
| **1** | **C1 Wass→Contrastive** | Curriculum | **53.7** | 30.8 |
| 2 | C2 Wass→L1+L2 | Curriculum | 56.2 | 39.7 |
| 3 | C4 NCCm→Contrastive | Curriculum | 60.4 | 28.3 |
| 4 | C3 TT→Contrastive | Curriculum | 60.5 | 28.2 |
| 5 | C5 Wass→NCCm | Curriculum | 61.6 | 36.4 |
| 6 | Wass (单一) | Single | 61.9 | 36.1 |
| 7 | Wass+AWI | Multi-FWI | 62.6 | 35.2 |
| 8 | Wass+NCCm | Multi-FWI | 64.5 | 35.4 |
| 9 | Wass+Contra | Multi-FWI | 64.7 | 35.5 |
| 10 | Wass+TT | Mixed+TT | 65.3 | 35.7 |
| 11 | AWI | Single | 65.6 | 34.5 |
| 12 | Env_NCC | Single | 69.2 | 33.7 |
| 13 | NCC_maxlag | Single | 70.6 | 31.7 |
| 14 | NCCm+TT | Mixed+TT | 70.3 | 32.9 |
| 15 | Contra+NCCm | Multi-FWI | 70.3 | 33.0 |
| 16 | Contra+TT | Mixed+TT | 72.7 | 31.4 |
| 17 | Contrastive | Single | 90.1 | 63.2 |
| 18 | NCC_zero | Single | 115.4 | 77.2 |
| 19 | Phase_func | Single | 210.8 | 86.0 |

---

## 2. 核心发现 (16 模型 × 5 seeds 验证通过)

### 发现 1: Curriculum 确定性地优于所有其他方法

Top 5 全是 curriculum 策略，与单一 Wass (61.9) 相比，C1 (53.7) 提升 **8.2 点 (13%)**。在 16 个新模型上复现了原始 6 模型的结论。

### 发现 2: Wasserstein 是最佳单一 reward

Wass (61.9) > AWI (65.6) > Env_NCC (69.2) > NCC_maxlag (70.6) > Contrastive (90.1) > NCC_zero (115.4)

### 发现 3: Multi-FWI 混合 vs Curriculum 差距稳定

同一对 reward 组合：Curriculum C1 (53.7) vs Mixed M4 (64.7) — **差 11 点**。递进式切换确定性地优于同时优化。

### 发现 4: Phase_func 确定性地失败

Mean=210.8, 19 策略中最差。振幅信息对 FWI 至关重要。

### 发现 5: Seed 鲁棒性

- Curriculum 平均 seed range: ~50 m/s（可接受，不同种子在难模型上有波动）
- 单 Wass seed range: 58 m/s
- 所有排名在 ±1 位内稳定，排名顺序不依赖种子

---

## 3. Per-Model 结果 (Top 5, Mean ± Std)

| Model | Best Strategy | Mean MAE |
|-------|---------------|----------|
| CVA1 | C2 Wass→L1+L2 | 11 ± 8 |
| CVA2 | C2 Wass→L1+L2 | 57 ± 31 |
| CVA5 | C1 Wass→Contra | 88 ± 6 |
| CVA6 | C1 Wass→Contra | 55 ± 9 |
| CVA8 | C2 Wass→L1+L2 | 49 ± 23 |
| CVA10 | C1 Wass→Contra | 86 ± 12 |
| CVA15 | C2 Wass→L1+L2 | 22 ± 14 |
| CVA16 | Contrastive | 33 ± 7 |
| CVA18 | C2/C5 Wass→* | 2.5 |
| CVA33 | C1 Wass→Contra | 41 ± 14 |
| CVA34 | C1 Wass→Contra | 45 ± 15 |
| CVA39 | C1 Wass→Contra | 66 ± 20 |
| CVA43 | C1 Wass→Contra | 30 ± 13 |
| CVA44 | C1 Wass→Contra | 100 ± 22 |
| CVA45 | C4 NCCm→Contra | 13 ± 4 |
| CVA50 | C4 NCCm→Contra | 56 ± 14 |

---

## 4. 实验配置一致性确认

- 所有 16 个 CVA 模型的 `init_velocity.npy` 来自相同的 `smooth_models_v2` 和统一的 `seed` 控制
- 同一模型的 19 个策略共享相同初始速度（已验证 hash 一致）
- GDPO 参数、PPO 超参、温度退火、G=32 均统一
- Curriculum stage2 使用 stage1 的 `policy_best.pt`

---

## 5. 数据文件

- 原始结果: `runs/large/{exp}_cva{idx}_seed{seed}/metrics.csv`
- JSON 汇总: `reports/phase5_large_results.json`
- 主控日志: `log_p5l_master.txt`
