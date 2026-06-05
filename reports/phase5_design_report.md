# Phase 5 Design Report — Robust Reward Functions for RL-FWI

**日期**: 2026-05-26  
**信息来源**: Wenzhe 的 `phase4_reward_design_report.md`（导师文献综述，含 8 篇核心文献及完整公式推导）  
**目标**: 将 reward 设计从 sample-wise 振幅残差转向 event-wise 相位/走时匹配

---

## 1. 动机

Phase IV 已证明：
- TT-only 最稳健但信息不足（仅初至）
- Contrastive 在困难模型上有效但没有显式惩罚 time-lag
- Envelope/Wasserstein/Windowed 从随机初值全面失败——现有实现没有体现文献关键设计

Phase 5 核心思路（来自导师文献）：用 cross-correlation、time-lag、AWI、OT 四类鲁棒 objective 构建 RL reward，利用 RL 不需要 adjoint source 的优势（argmax、窗口选择、置信权重可以直接用）。

---

## 2. 实验矩阵

| 编号 | Reward | 类型 | 新增/修复 | 文献依据 |
|------|--------|------|----------|---------|
| R0 | Wasserstein_abs | OT | 修复 | Métivier et al. 2016; Wei et al. 2026 |
| R1 | NCC_zero | Cross-correlation | 新增 | Liu et al. 2017 |
| R2 | NCC_maxlag | Cross-correlation + lag | 新增 | Zhang et al. 2019 |
| R3 | Envelope_NCC | Envelope + NCC | 新增 | Oh & Alkhalifah 2018 |
| R4 | AWI | Matching filter | 新增 | Warner & Guasch 2016 |

**Phase IV 基线**（已有，无需重跑）: TT-only, Contrastive, Prog_TT→L2, L1+L2

---

## 3. Reward 函数设计细节

### 3.1 R0: Wasserstein_abs（CDF-based Time-Domain W₁）

**文献**: Métivier et al. 2016 "Measuring the misfit between seismograms using an optimal transport distance"; Wei et al. 2026

**设计思路**: 
将地震信号视为时间轴上的非负密度分布，计算 CDF 差值的积分作为 W₁ 距离。相比 L2，W₁ 对时间平移是凸的，能缓解 cycle-skipping。

**计算步骤**（per trace）:

```
输入: d_pred[t], d_obs[t]  (长度 nt)

Step 1 — 非负化:
    d⁺[t] = |d[t]|

Step 2 — 归一化为概率分布:
    p[t] = d⁺[t] / Σₜ d⁺[t]   (clamp denominator to ≥ 1e-10)

Step 3 — 累积分布函数:
    C[t] = Σ_{i=0}^{t} p[i]

Step 4 — W₁ 距离:
    W₁ = Σₜ |C_pred[t] - C_obs[t]|

Step 5 — Per-group 聚合:
    R_wass = -Σ_{traces} W₁(trace)
```

**为什么修复前是错的**:
旧实现对 trace 做 `.sort()` 再比较排序值差异——这是振幅空间的 W₁，摧毁了全部时间结构。两个时间错位的相同信号 W₁=0。

**关键特性**:
- 对时间平移凸（不会 cycle-skip）
- 不要求振幅匹配（只比较能量在时间轴上的分布）
- abs() 非负化保留了包络信息

**可选非负化**: `abs`（默认）| `square` | `envelope`

---

### 3.2 R1: NCC_zero（Normalized Zero-Lag Cross-Correlation）

**文献**: Liu et al. 2017 "Robust time-domain full waveform inversion with normalized zero-lag cross-correlation objective function", GJI

**设计思路**:
不再强迫逐点振幅相等（L2），而是强调相位和形态相似。零时移归一化互相关天然对振幅误差、震源强度误差稳健。

**计算步骤**（per trace）:

```
输入: d_pred[t], d_obs[t]  (长度 nt，已 zero-mean)

Step 1 — 零时移互相关:
    cc₀ = Σₜ d_obs[t] · d_pred[t]

Step 2 — 归一化:
    norm_pred = √(Σₜ d_pred[t]²)
    norm_obs  = √(Σₜ d_obs[t]²)
    ncc₀ = cc₀ / (norm_pred · norm_obs + ε)

Step 3 — Per-group 聚合（均值）:
    R_ncc = (1/N_traces) · Σ_{traces} ncc₀(trace)
```

**关键特性**:
- 取值范围 [-1, 1]，1 = 完全正相关，0 = 不相关，-1 = 完全反相关
- 对整体振幅缩放不敏感（归一化消除了振幅差异）
- 实现最简单，计算成本最低
- 局限：对 cycle-skipping 仍然敏感（信号错位超过半个周期时 ncc₀ 可能仍然低）

**与 L2 的对比**:
- L2: 问"每个采样点的振幅是否一样" → 对振幅误差敏感
- NCC_zero: 问"波形的相位和形态是否相似" → 对振幅误差稳健

---

### 3.3 R2: NCC_maxlag（Maximum NCC + Lag Penalty）

**文献**: Zhang et al. 2019 "Normalized nonzero-lag crosscorrelation elastic full-waveform inversion", Geophysics

**设计思路**:
从 zero-lag 扩展到 nonzero-lag：允许波形有一定时间偏移，在 ±τ_max 范围内搜索最大互相关。同时加入 lag penalty 鼓励最终对齐到零时移。

**计算步骤**（per trace）:

```
输入: d_pred[t], d_obs[t]  (长度 nt，已 zero-mean)
参数: τ_max = 80 (最大搜索时移，单位 samples)

Step 1 — 滑动互相关:
    对 τ ∈ [-τ_max, τ_max]:
        cc(τ) = Σₜ d_obs[t+τ] · d_pred[t]  (zero-pad 边界)
        ncc(τ) = cc(τ) / (√(Σ d_obs[t+τ]²) · √(Σ d_pred[t]²) + ε)

Step 2 — 找到最大相关峰:
    ncc_max = max_τ ncc(τ)
    τ* = argmax_τ ncc(τ)

Step 3 — Lag penalty（鼓励零时移）:
    penalty = λ · |τ*|
    λ 建议: 0.001 ~ 0.01

Step 4 — Per-trace reward:
    R_trace = ncc_max - penalty

Step 5 — Per-group 聚合:
    R_ncc_maxlag = (1/N_traces) · Σ_{traces} R_trace
```

**关键特性**:
- ncc_max 衡量"最佳对齐后的波形相似度"（≥ ncc₀）
- lag penalty 显式惩罚走时偏差，驱动模型向正确方向更新
- 相比 TT-only（仅初至），nonzero-lag NCC 利用了全波形信息
- 滑动互相关通过 FFT 加速：cc = IFFT(FFT(d_obs)* · FFT(d_pred))

**与 Phase IV Contrastive 的对比**:
- Contrastive: spectrum similarity + trace CC max（全局相似度），没有 lag penalty
- NCC_maxlag: 在时间域显式计算并惩罚 time-lag，更直接

---

### 3.4 R3: Envelope_NCC（Envelope-Based Global Correlation Norm）

**文献**: Oh & Alkhalifah 2018 "Full waveform inversion using envelope-based global correlation norm", GJI

**设计思路**:
相位 IV 的 Envelope L2 失败了——因为仍然在比包络振幅。Envelope NCC 用包络的归一化互相关替代 L2，既保留了包络产生人工低频的优势，又避免了振幅尺度问题。

**计算步骤**（per trace）:

```
输入: d_pred[t], d_obs[t]  (长度 nt)

Step 1 — Hilbert 包络:
    e_pred[t] = |Hilbert(d_pred[t])|
    e_obs[t]  = |Hilbert(d_obs[t])|

Step 2 — 去均值:
    e_pred -= mean(e_pred)
    e_obs  -= mean(e_obs)

Step 3 — 零时移 NCC:
    cc₀ = Σₜ e_obs[t] · e_pred[t]
    ncc_env = cc₀ / (‖e_pred‖ · ‖e_obs‖ + ε)

Step 4 — Per-group 聚合:
    R_env_ncc = (1/N_traces) · Σ_{traces} ncc_env(trace)
```

**为什么应该比 Envelope L2 好**:
- Envelope L2: 罚 ‖e_pred - e_obs‖² → 对包络振幅敏感，初始模型差时信号太弱 → reward 区分度低
- Envelope NCC: 只比包络形态 → 即使信号弱，只要形状对就能得高分

**关键特性**:
- 包络产生人工低频 → 帮助恢复长波长背景速度
- NCC 归一化 → 对振幅不敏感
- 两者结合 = Oh & Alkhalifah 2018 的两阶段策略的第一步

---

### 3.5 R4: AWI（Adaptive Waveform Inversion — Matching Filter Spread）

**文献**: Warner & Guasch 2016 "Adaptive waveform inversion: Theory", Geophysics

**设计思路**:
不直接比较 d_pred 和 d_obs，而是求一个 matching filter w，使得 w ∗ d_pred ≈ d_obs。如果速度模型正确，w 应该接近零时移 δ 函数。Reward 惩罚 w 的时间散布程度。

**计算步骤**（per trace）:

```
输入: d_pred[t], d_obs[t]  (长度 nt，已 zero-mean)
参数: ε = 1e-3 (water-level regularization)

Step 1 — 频域 Wiener filter:
    D_pred(ω) = FFT(d_pred)
    D_obs(ω)  = FFT(d_obs)
    
    W(ω) = conj(D_obs(ω)) · D_pred(ω) / (conj(D_obs(ω)) · D_obs(ω) + ε · max|D_obs|²)

Step 2 — 时域 filter:
    w[t] = IFFT(W(ω))

Step 3 — 时间散布（加权二阶矩）:
    计算 w 的"重心"偏离零时移的程度:
    
    τ_center = Σₜ t · w[t]² / Σₜ w[t]²            (重心)
    τ_spread² = Σₜ (t - τ_center)² · w[t]² / Σₜ w[t]²   (散布)
    
    （零时移在索引 0 处，负索引对应负时移）

Step 4 — Reward:
    R_awi = -τ_spread（散布越小越好）
    
    或组合形式:
    R_awi = -(α · |τ_center| + β · τ_spread)
```

**简化版（AWI_L1）**:
```
    直接用加权一阶矩（文档 §5.3 形式）:
    R_awi = -Σₜ |t| · w[t]² / Σₜ w[t]²
```
这个形式不区分重心偏移和散布，但实现更简单。

**关键特性**:
- 不直接比较波形 → 对振幅差异和波形差异都更稳健
- 如果速度模型完美，w ≈ δ(t)，spread ≈ 0
- 如果速度模型有走时误差，w 的能量会偏离零时移
- 比 NCC 更复杂，但理论上更能处理波形形变

**两种实现方案**:
| 方案 | 计算 | 优点 | 缺点 |
|------|------|------|------|
| AWI_full | Wiener filter + 重心 + 散布 | 理论完备 | 需要两个统计量 |
| AWI_L1 | Wiener filter + 加权一阶矩 | 实现简单 | 混合了偏移和散布 |

建议先用 AWI_L1 快速验证概念，有效再升级到 AWI_full。

---

## 4. 实现状态

| Reward | 代码位置 | 状态 |
|--------|---------|------|
| R0 Wasserstein_abs | `agents/fwi_rewards.py` | ✅ 已修复，训练中 |
| R1 NCC_zero | 待实现 | 📋 |
| R2 NCC_maxlag | 待实现 | 📋 |
| R3 Envelope_NCC | 待实现 | 📋 |
| R4 AWI | 待实现 | 📋 |

---

## 5. 测试验证计划

每个新 reward 需要验证：

1. **基本正确性**: 相同信号 → reward 最大；随机信号 → reward 接近 0
2. **时间敏感性**: 时间平移后 reward 应下降（证明对走时敏感）
3. **振幅无关性**: 振幅缩放后 reward 不变（NCC 类）
4. **梯度流**: reward 可微（用于 PPO 优势计算，非严格需要但需要稳定数值）
5. **计算效率**: 对 G=32, n_shots=5, nt=1000, nr=70 的典型 batch，单次 reward 计算 < 1s

---

## 6. 实验执行计划

### Phase 5A: 单 Reward 消融

**模型**: CVA18, CVA50, CVA10, CVA6, CVA8, CVA5（6 个，与 Phase IV 一致）

**配置**: 5000 steps, G=32, ppo_epochs=4, lr=5e-3（与 Phase IV 一致，确保可比性）

| 优先级 | Reward | 预计启动 |
|--------|--------|---------|
| 🔴 运行中 | R0 Wasserstein_abs | 已启动（GPU 1,2,3） |
| 🟡 待实现 | R1 NCC_zero | 实现+测试后 |
| 🟡 待实现 | R2 NCC_maxlag | 实现+测试后 |
| 🟡 待实现 | R3 Envelope_NCC | 实现+测试后 |
| 🟢 可选 | R4 AWI | 前三组有正结果后 |

### Phase 5B: Curriculum 组合

Phase 5A 结束后，选 Top 2-3 reward + Phase IV 最佳 baseline，设计递进 curriculum（参考 phase4_reward_design_report.md §6 Stage B）。

---

## 7. 预期贡献

Phase 5 完成后，论文实验部分将包含：

1. **Phase IV** — 9 策略基准（L2, L1+L2, TT-only, Contrastive, Envelope, Windowed, Prog_TT→L2, Multi_TT+L2, Wasserstein_old）
2. **Phase 5A** — 5 个文献驱动的鲁棒 reward（Wasserstein_fixed, NCC_zero, NCC_maxlag, Envelope_NCC, AWI）
3. **Phase 5B** — Curriculum 组合策略

核心叙事：L2 waveform misfit → cycle-skipping → 鲁棒 objective 在 RL 框架中的优势（不需要 adjoint source）

---

## 参考文献

- Liu et al. 2017, GJI — normalized zero-lag CC
- Zhang et al. 2019, Geophysics — normalized nonzero-lag CC elastic FWI
- Oh & Alkhalifah 2018, GJI — envelope-based global correlation norm
- Warner & Guasch 2016, Geophysics — adaptive waveform inversion
- Métivier et al. 2016 — optimal transport for FWI
- Zhang et al. 2018, SEG — CGG time-lag FWI
- Luo & Schuster 1991, Geophysics — wave-equation traveltime inversion
