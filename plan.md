# Multi-Reward Seismic Inversion via Reinforcement Learning — 实验规划

**阶段**: Phase IV | **更新**: 2026-05-18

---

## 0. 核心目标

提出一个 **multi-reward RL 地震反演框架**：
- 不可微正演 → 物理 reward 驱动优化
- 递进式 reward 设计：走时 → 全波形 → 多物理场
- 论证方法的可行性和局限性

---

## 1. 实验设定

### 1.1 采集几何：透射式

```
        检波器排列 (z=0, 地表)
        ●  ●  ●  ...  ●  (70个)
        |  |  |       |
    速度模型 (70×70 grid)
        |  |  |       |
        ★  ★  ★  ...  ★  (5个震源, z=底部)
        震源排列 (z=nz-1, 模型底部)
```

模拟天然地震场景：震源在深部，检波器在地表。

### 1.2 速度模型：B-spline 平滑

```
CVA 原始模型 (70×70)
    ↓ bspline2d_inverse → 4×4 控制点
    ↓ bspline2d_prolong → 70×70 平滑速度模型 (作为"真实模型")

优势：保证 B-spline 参数空间内存在精确解
```

### 1.3 参数化

- **控制点网格**：4×4 = 16 控制点，后期扩展到 6×6
- **策略**：128 参数 Gaussian 分布 (μ + σ per control point)
- **搜索空间**：16 维 B-spline 控制点空间

---

## 2. 递进式 Reward 设计

| 阶段 | Reward | 目标 | 状态 |
|------|--------|------|------|
| R1 | **走时 (Travel-time)** | 初至时间差最小化 → 长波长背景速度 | 📋 |
| R2 | **全波形 (FWI)** | Sign-preserving log L1+L2 data misfit | 📋 |
| R3 | **多物理场** | FWI + 地质先验 + SI imaging energy | 📋 |

### 2.1 走时 Reward

```python
R_tt = -|t_first(p_pred) - t_first(p_obs)|  # 初至时间差的负绝对值
```

目的：先抓住背景速度趋势（长波长），避免 cycle-skipping。

### 2.2 全波形 Reward

```python
R_fwi = -L1(sign_log(p_pred), sign_log(p_obs)) - L2(sign_log(p_pred), sign_log(p_obs))
```

在走时基础上精细化反演。

### 2.3 多 Reward 融合

```python
A = w_tt * group_norm(R_tt) + w_fwi * group_norm(R_fwi) + w_pr * group_norm(R_prior)
A ← batch_norm(A)
```

GDPO 解耦各 reward 分量的量级差异。

---

## 3. Baseline 对比

| 方法 | 说明 |
|------|------|
| PPO + 单 reward (FWI) | 消融多 reward 收益 |
| PPO + 递进 reward | 主方法 |
| CMA-ES + 递进 reward | 优化器无关性验证 |
| 梯度 FWI (可微) | 可微 vs 不可微分界线 |

---

## 4. 实验矩阵

### 4.1 模型选择

从 CVA 测试集 (files 44-59) 选取 4 个模型，B-spline 平滑后作为真实模型。

| 模型 | CVA 文件 | 说明 |
|------|---------|------|
| M1 | CVA[50] | 简单曲界面 |
| M2 | CVA[52] | 中等复杂度 |
| M3 | CVA[55] | 较复杂 |
| M4 | CVA[58] | 较复杂 |

### 4.2 实验矩阵

| # | Reward | 策略 | 几何 | 模型 | 状态 |
|---|--------|------|------|------|------|
| E1 | 走时 only | PPO | 透射 | M1-M4 | 📋 |
| E2 | FWI only | PPO | 透射 | M1-M4 | 📋 |
| E3 | 走时→FWI 递进 | PPO | 透射 | M1-M4 | 📋 |
| E4 | 走时→FWI+Prior | PPO | 透射 | M1-M4 | 📋 |
| E5 | FWI only | CMA-ES | 透射 | M1 | 📋 |
| E6 | 走时→FWI | CMA-ES | 透射 | M1 | 📋 |

---

## 5. 代码改动

```
需要新增/修改：
├── agents/traveltime_reward.py    # NEW: 走时 reward 计算
├── agents/transmission_forward.py # MODIFY: 震源在底部
├── train_rl_fwi.py                # MODIFY: 支持递进 reward 切换
└── train_cmaes_latent.py          # MODIFY: 适配 B-spline 空间 + 递进 reward
```

---

## 6. 记录规范

每个实验保存：
- `config.json` — 完整参数
- `metrics.csv` — step 级指标
- `progression.png` — 初始/最终/真值对比
- `summary_step_*.png` — 定期快照
- `best_velocity.npy` — 最优速度模型
- 结果追加到 `memory-bank/progress.md`
