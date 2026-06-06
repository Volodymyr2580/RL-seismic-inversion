# Phase IV — Multi-Reward Seismic Inversion

**开始**: 2026-05-18 | **几何**: 透射式 | **模型**: CVA B-spline 平滑

---

## 实验记录

### 待完成

| # | 实验 | 状态 |
|---|------|------|
| 1 | 走时 reward 实现 | 📋 |
| 2 | 透射几何适配 (震源底部) | 📋 |
| 3 | B-spline 平滑模型生成 | 📋 |
| 4 | E1: 走时-only PPO (M1-M4) | 📋 |
| 5 | E2: FWI-only PPO (M1-M4) | 📋 |
| 6 | E3: 走时→FWI 递进 PPO (M1-M4) | 📋 |
| 7 | E4: 走时→FWI+Prior PPO (M1-M4) | 📋 |
| 8 | E5: FWI-only CMA-ES (M1) | 📋 |
| 9 | E6: 走时→FWI CMA-ES (M1) | 📋 |

---

## 已完成 Phase I-III

详见 `reports/` 文件夹。

---

## Phase VI — Corrected Single-Reward NCC Convergence Check

**启动准备**: 2026-06-06
**目的**: 在修复 reward 张量布局后，重新检查 Cross Correlation / NCC 类单 reward 在 CVA B-spline 重建模型上的 RL 收敛表现。

### 本轮关键修复

- 统一 `fwi_rewards.py` 的输入布局为 `[G, shot, receiver, time]` / `[shot, receiver, time]`。
- 修复受影响的 trace-wise reward: `windowed_l2`, `wasserstein`, `wasserstein_w2`, `contrastive`, `ncc_zero`, `ncc_maxlag`, `envelope_ncc`, `awi`, `phase_func`。
- 修复 `traveltime_reward.py` 的同类布局问题。
- 新增 `tests/test_reward_sanity.py`，用于验证 NCC 的振幅缩放不变性、时间平移敏感性、`NCC_maxlag` shift recovery 和 canonical layout。

### Phase6 默认实验设置

- Reward: `ncc_maxlag` 单 reward
- Seed: `42`
- Models: `1 2 5 6 8 10 15 16 18 50`
- 数据源: `data/smooth_models_v2` 中的 CVA B-spline 平滑模型
- 训练参数: 沿用 Phase5 单 reward 设置，`G=32`, `steps=5000`, `ppo_epochs=4`, `lr=5e-3`, transmission geometry
- 输出目录: `runs/phase6/FWI_ncc_maxlag_cva{idx}_seed42`

### 可视化要求

- 四个模型: initial, true, best MAE, final converged
- 残差: `|best - true|`, `|final - best|`
- 曲线: reward curve, MAE convergence
- 炮集: 每炮 wiggle 红黑对比，black=true, red=initial/best/final candidate
