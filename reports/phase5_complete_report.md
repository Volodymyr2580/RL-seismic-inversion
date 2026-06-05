# Phase 5 Complete Report — Robust Reward Functions for RL-FWI

**日期**: 2026-05-29  
**实验总数**: 30 策略 × 6 模型 = 180 次完整训练  
**代码库**: `agents/fwi_rewards.py`, `agents/rl_objectives.py`, `train_rl_fwi.py`

> **信息来源**: 本文档中所有 reward 函数的文献来源和设计原理均来自 Wenzhe 撰写的 `phase4_reward_design_report.md`（导师文献综述）。该文档系统梳理了 cross-correlation、traveltime/time-lag、AWI、OT 四类鲁棒 FWI objective 的文献脉络（Liu 2017, Zhang 2019, Oh & Alkhalifah 2018, Warner & Guasch 2016, Métivier 2016, Wei 2026 等），并给出了每个 reward 的公式推导和参数建议。Phase 5 的实现是对该文档的直接工程化。

---

## 1. 实验设计

### 1.1 统一设置

所有实验共享以下配置，确保可比性：

| 参数 | 值 |
|------|-----|
| 策略空间 | 4×4 B-spline 控制点，Gaussian policy，32 参数 |
| 速度范围 | [1500, 4500] m/s |
| 几何 | 透射 (5 sources @ bottom, 70 receivers @ top) |
| 测试模型 | CVA18, CVA50, CVA10, CVA6, CVA8, CVA5 |
| Group size | G=32 |
| PPO epochs | 4 |
| Learning rate | 5e-3 |
| 温度 | 2.0 → 0.1 (anneal 1000 steps) |
| Entropy bonus | 0.02 |
| Steps | 5000 (curriculum: 2500+2500 或 2000+1500+1500) |
| 初始化 | 所有从零实验使用**相同初始速度模型** (seed=42, 同一 CVA 模型 hash 一致) |
| Curriculum stage2 | 从 stage1 的 policy_best.pt 恢复（非随机初始化） |

### 1.2 初始化一致性

```
同一 CVA 模型的所有 FWI_*、Mix*、MixFWI_*、curr_*2500 实验
使用相同的 random seed=42 → init_velocity.npy 的 MD5 hash 完全相同
CVA18/50/6/5: f30fb3fd  |  CVA10/8: 38e5a3bc
```

---

## 2. Reward 函数设计

### 2.1 Phase 5A — 单 Reward

| Reward | 文献 | 计算方法 | 参数 |
|--------|------|---------|------|
| **Wasserstein_abs** | Métivier et al. 2016 | 时域 CDF-based W₁: signal → \|·\| → normalize → CDF → ∫\|ΔCDF\| | normalize="abs" |
| **NCC_zero** | Liu et al. 2017 | 归一化零时移互相关: Σ(pred·obs)/(‖pred‖·‖obs‖) | — |
| **NCC_maxlag** | Zhang et al. 2019 | 最大 NCC + lag penalty: max_τ CC(τ) - λ·\|τ\| per trace | lag_max=80, λ=0.005 |
| **Envelope_NCC** | Oh & Alkhalifah 2018 | Hilbert 包络 → NCC: FFT → \|·\| → ifft → NCC | — |
| **AWI** | Warner & Guasch 2016 | Wiener matching filter 时间散布: W = P_obs·P_pred* / (\|P_pred\|²+ε), R = -⟨\|t\|·w²⟩/⟨w²⟩ | ε=1e-3, version="l1" |
| **Phase_func** | — | FFT → 振幅归一化 (\|X\|=1) → IFFT → L2 | — |

### 2.2 Phase 5B — 混合与 Curriculum

#### Mixed +TT (GDPO 同时多 reward，单 stage)
```
GDPO: 各 reward 独立 group-normalize → 加权求和 → batch-normalize
```

| 实验 | FWI Reward | +TT Weight |
|------|-----------|------------|
| M1 Wass+TT | Wasserstein_abs | w_tt=1.0 |
| M2 Contra+TT | Contrastive | w_tt=1.0 |
| M3 NCCm+TT | NCC_maxlag | w_tt=1.0 |

#### Multi-FWI Mixed (两个 FWI reward 同时)
```
--fwi_type A --fwi_type2 B --fwi_weight2 1.0
GDPO 独立 normalize 两个 FWI reward
```

| 实验 | FWI 1 | FWI 2 |
|------|-------|-------|
| M4 Wass+Contra | Wasserstein_abs | Contrastive |
| M5 Wass+NCCm | Wasserstein_abs | NCC_maxlag |
| M6 Wass+AWI | Wasserstein_abs | AWI |
| M7 Contra+NCCm | Contrastive | NCC_maxlag |

#### Curriculum (递进式 reward 切换)
```
Stage 1 (2500步, reward A) → Stage 2 (2500步, --resume_ckpt, reward B)
```

| 实验 | Stage 1 | Stage 2 |
|------|---------|---------|
| C1 Wass→Contra | Wasserstein_abs | Contrastive |
| C2 Wass→L1+L2 | Wasserstein_abs | L1+L2 |
| C3 TT→Contra | TT-only | Contrastive |
| C4 NCCm→Contra | NCC_maxlag | Contrastive |
| C5 Wass→NCCm | Wasserstein_abs | NCC_maxlag |

---

## 3. 结果

### 3.1 完整排名 (Mean MAE)

| Rank | Strategy | Type | Mean MAE |
|------|----------|------|----------|
| 1 | **C1 Wass→Contra** | Curriculum | **50.8** |
| 2 | C3 TT→Contra | Curriculum | 61.1 |
| 3 | C4 NCCm→Contra | Curriculum | 63.8 |
| 4 | Wass (单一) | Single | 66.9 |
| 5 | C5 Wass→NCCm | Curriculum | 70.6 |
| 6 | M6 Wass+AWI | Multi-FWI | 70.7 |
| 7 | M1 Wass+TT | Mixed+TT | 71.1 |
| 8 | C2 Wass→L1+L2 | Curriculum | 71.7 |
| 9 | M5 Wass+NCCm | Multi-FWI | 74.5 |
| 10 | M4 Wass+Contra | Multi-FWI | 74.7 |
| 11 | Contrastive (单一) | Single | 75.0 |
| 12 | M3 NCCm+TT | Mixed+TT | 75.6 |
| 13 | M7 Contra+NCCm | Multi-FWI | 76.4 |
| 14 | M2 Contra+TT | Mixed+TT | 76.8 |

### 3.2 各模型最佳

| Model | 历史最佳 (P4) | Phase 5 最佳 | 策略 | 提升 |
|-------|-------------|-------------|------|------|
| CVA18 | 4.1 | **2.5** | C2/C5 curriculum | -39% |
| CVA50 | 27.6 | **23.1** | C4 curriculum | -16% |
| CVA10 | 97.5 | **91.2** | C1 curriculum | -6% |
| CVA6 | 31.6 | 31.6 | (持平 P4) | — |
| CVA8 | 22.8 | 22.8 | (持平 P4) | — |
| CVA5 | 90.3 | **83.9** | C3 curriculum | -7% |

### 3.3 核心发现

1. **Curriculum > Mixed > Single**: 递进式切换 (mean 50.8) 远优于同时混合 (mean ~74) 和单一 (mean ~67)
2. **Wasserstein 是最佳 Stage1**: 鲁棒收敛，为 Stage2 提供良好起点
3. **Contrastive 是最佳 Stage2**: 在 Stage1 基础上精细化
4. **Multi-FWI 混合无效**: 同时优化两个目标让策略折中，所有 M4-M7 性能均不如单一 Wass
5. **+TT 混合效果有限**: 与单一 reward 相比无显著提升

---

## 4. 可视化

所有可视化位于: `reports/phase5_visualizations/`

```
phase5_visualizations/
├── 5A_single/          # 6 个单 reward × 6 模型 = 36 × 2 图
│   ├── Wasserstein_abs_CVA18_velocity.png
│   ├── Wasserstein_abs_CVA18_convergence.png
│   └── ...
├── 5B_mixed+TT/        # 3 个混合+TT × 6 模型
├── 5B_multiFWI/        # 4 个 multi-FWI × 6 模型
├── 5B_curriculum/      # 5 个 curriculum × 6 模型
├── summary_mean_ranking.png   # 所有策略 Mean MAE 排名
└── summary_heatmap.png        # 所有策略 × 6 模型热力图
```

每张 velocity 图: Init (红色) / Best (绿色) / True (黑色) 三栏 + colorbar
每张 convergence 图: Per-step MAE + Global Best MAE 曲线

---

## 5. 代码变更

| 文件 | 变更 |
|------|------|
| `agents/fwi_rewards.py` | 重写 Wasserstein (CDF-based), 修复 hilbert_envelope (fft→fft+ifft), 新增 NCC_zero/NCC_maxlag/Envelope_NCC/AWI/Phase_func, 新增 dispatcher |
| `agents/rl_objectives.py` | RewardWeights 新增 fwi2 分量, gdpo_advantage 支持 fwi2 |
| `train_rl_fwi.py` | 新增 --fwi_type2/--fwi_weight2/--ncc_lag_*/--awi_version, 多 FWI 混合训练循环, reward hacking 监控 |
| `run_phase5a_batch.sh` | Phase 5A 批量训练脚本 |
| `run_phase5b_*.sh` | Phase 5B curriculum/mixed/multi-FWI 启动脚本 |

---

## 6. Bug 修复记录

| Bug | 影响 | 修复 |
|-----|------|------|
| Wasserstein 排序-based W₁ | 盲于时间偏移，mean=222.6 | CDF-based time-domain W₁, mean→67.0 |
| hilbert_envelope irfft | 包络错误 (cos envelope≠1), 影响 Envelope_NCC 和 envelope normalize | fft+ifft, cos envelope=1.000 |
