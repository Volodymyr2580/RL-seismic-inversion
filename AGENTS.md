# AGENTS.md – 核心组件说明

项目：多目标 RL 引导的单观测地震反演（Phase II）
更新：2026-05-14

---

## 核心 Pipeline

```
p_data [N_s, N_r, N_t]   (单一观测数据)
    │
    ▼
┌──────────────────────────┐
│  1. CNN Encoder           │  seismic → 特征 → Beta(α, β) 分布参数
│     (~50K params)         │  输出: α[4×4], β[4×4]
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  2. Beta 策略采样         │  v_ctrl ~ Beta(α, β) → 缩放到 [v_min, v_max]
│     采样 G 组控制点        │  输出: v_ctrl [G, 4, 4]
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  3. B-spline 速度重建     │  4×4 控制点 → 双三次插值 → 70×70 速度模型
│                            │  输出: v_model [G, 70, 70]
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  4. Deepwave 正演模拟     │  速度模型 → 波动方程 → 合成炮集
│     (不可微，仅 reward 用)  │  输出: p_pred [G, N_s, N_r, N_t]
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  5. 多目标 Reward 计算    │
│    R_fwi:  data misfit    │  sign-preserving log L1+L2
│    R_si:   SI 成像能量     │  RTM focusing energy (正)
│    R_prior: 地质先验       │  平滑性 + 单调性 + 范围约束
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  6. GDPO-Guard 策略优化   │
│    GDPO: per-reward 解耦  │  各自组归一化 → 加权求和 → batch 标准化
│    Guard: ratio correction│  监控 ratio mean/std, clip fraction
│    Clipped surrogate loss │  decoupled ε_low, ε_high
└──────────────────────────┘
```

---

## 组件详解

### 1. CNN Encoder（`agents/cnn_encoder.py`）

轻量 CNN，从炮集数据提取特征并输出 Beta 分布参数。

- **输入**: `p_data` [B, N_s, N_r, N_t] — 炮集（可 reshape 为 2D 卷积输入）
- **输出**: `alpha` [B, 4, 4], `beta` [B, 4, 4] — 每个控制点的 Beta 分布参数
- **架构**: 3 层 Conv2D + BatchNorm + SiLU → AdaptiveAvgPool → 双头 1×1 Conv
- **参数量**: ~50K

### 2. Beta 策略（`agents/beta_policy.py`）

有界连续 B-spline 控制点的 Beta 分布采样。

- **采样**: `u ~ Beta(α, β)` → `v_ctrl = v_min + (v_max - v_min) * u`
- **log_prob**: 包含 Beta 分布对数密度（无需 Jacobian 修正，Beta 天然在 [0,1] 上）
- **温度控制**: `α' = α / T, β' = β / T` 调节探索程度
- **变体**:
  - `BetaSplinePolicy`: CNN 条件化版本
  - `LearnableBetaSplinePolicy`: 无 encoder 的可学习参数版本（基线）

### 3. B-spline 速度重建（`agents/bspline.py`, `agents/velocity_reconstructor.py`）

- **功能**: `bspline2d_prolong`: 4×4 控制点 → 70×70 速度模型（双三次 B-spline）
- **逆变换**: `bspline2d_inverse`: 70×70 速度模型 → 4×4 控制点（用于 CVA 预训练投影）
- **固定层**: 支持固定上覆水层/近地表层速度

### 4. 正演模拟器（`agents/forward_simulator.py`）

- **实现**: `deepwave.scalar` 标量波动方程
- **配置**: 与 OpenFWI Vel family 几何对齐（dx=10m, dt=0.001s, freq=15Hz）
- **关键**: **不要求可微**——只用于计算 reward 信号
- **限制**: deepwave 不支持 velocity-model 级 batch，G 个模型需串行正演

### 5. 奖励计算器（`agents/reward_calculator.py`）

| Reward | 公式 | 方向 |
|--------|------|------|
| `R_fwi` | `-L1-L2(sign_log(p_pred), sign_log(p_obs))` | 最大化（loss 取负） |
| `R_si` | `Σ(RTM_image)²` | 最大化（聚焦能量） |
| `R_prior` | `-w₁∇²v - w₂ReLU(-∂v/∂z) - w₃bound_pen` | 最大化（惩罚取负） |

### 6. GDPO-Guard 优化器（`agents/rl_objectives.py`）

**GDPO 部分**（已有 `gdpo_advantage`）:
```python
A_i = w_fwi * group_norm(R_fwi)
    + w_si  * group_norm(R_si)
    + w_pr  * group_norm(R_prior)
A_i ← batch_norm(A_i)  # 跨 batch 标准化防止量级漂移
```

**GRPO-Guard 部分**（需新增）:
- Ratio 诊断：记录 `ratio_mean`, `ratio_std`, `clip_fraction`
- Ratio 修正：当 `|ratio_mean - 1.0| > threshold` 时施加软偏移
- Decoupled clipping：`ε_low`（鼓励探索）、`ε_high`（限制利用）

**Clipped Policy Loss**（已有 `clipped_policy_loss`）:
```python
L = -min(exp(log_ratio)*A, clip(exp(log_ratio), 1-ε_low, 1+ε_high)*A)
```

### 7. CVA 预训练（`pretrain_cnn_cva.py`）

- 使用 CVA/CurveVel_A 数据集（`data/model` `.npy` 文件）
- `bspline2d_inverse(v_true)` → 控制点标签
- CNN MSE 预训练：`p_data → v_ctrl`
- 输出 checkpoint 供 RL 阶段加载为 warm start
- **约束**: 测试模型不得在预训练数据集中出现

---

## 备选/基线模块（保留但不作为主线）

| 模块 | 定位 | 文件 |
|------|------|------|
| ViT 离散策略 | DeepWaveRL baseline | `agents/policy_network.py` |
| Gaussian 连续策略 | Beta 分布消融 | `agents/continuous_policy.py` |
| `train_single_observation_rl.py` | Phase I 入口（参考） | 保留 |
| `train_DAPO.py` | Phase I DAPO baseline（参考） | 保留 |

---

## 工作约束

- 禁止批量删除文件或目录
- 需要删除文件时，一次只删除一个明确路径的文件
- 每次实验后更新 `memory-bank/progress.md`
- Git commit 格式: `<type>(<scope>): <summary>`

---

## 训练入口

| 入口 | 阶段 | 用途 |
|------|------|------|
| `pretrain_cnn_cva.py` | 预训练 | CVA 数据集 CNN Beta 策略预训练 |
| `train_rl_fwi.py` | RL 主实验 | 单观测多目标 RL 反演（Phase II 主入口） |
