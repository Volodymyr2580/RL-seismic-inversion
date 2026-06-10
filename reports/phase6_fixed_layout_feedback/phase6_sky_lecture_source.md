# Phase6 Fixed-Layout 单 Reward 实验讲义

> 本讲义面向导师汇报，目标不是只展示图片，而是解释 Phase6 fixed-layout 之后各类 reward 的真实排序、Cross Correlation 的当前状态，以及下一步为什么应该进入 windowed time-lag NCC。

## 1. 本轮实验回答的问题

Phase6 的核心问题是：在统一 seismic tensor layout 后，单 reward 是否能稳定引导 B-spline RL 反演收敛？

这次我们把所有 trace-wise reward 的输入统一为 canonical layout：

```math
p_{\text{pred}} \in \mathbb{R}^{G \times N_s \times N_r \times N_t},
\qquad
p_{\text{obs}} \in \mathbb{R}^{N_s \times N_r \times N_t}
```

其中最后一维才是时间轴。这个约束非常关键，因为 W1/W2/NCC/AWI/TT 都是沿时间轴逐 trace 计算的 reward。

实验矩阵如下：

| 项目 | 设置 |
|---|---|
| Reward | `l1l2`, `tt_only`, `wasserstein`, `wasserstein_w2`, `ncc_zero`, `ncc_maxlag`, `envelope_ncc`, `awi` |
| Models | CVA `1, 2, 5, 6, 8, 10, 15, 16, 18, 50` |
| Seed | 42 |
| Policy | Gaussian B-spline policy, `G=32`, `ppo_epochs=4`, `lr=5e-3` |
| Geometry | Transmission |
| Output per run | 四模型图、残差图、reward/MAE 曲线、shot gather wiggle overlay |

## 2. 我们如何读这些图

每个 run 的 `models_residuals_curves.png` 包含：

- initial model
- true model
- best MAE model
- final converged model
- best MAE model 和 true model 的残差
- final model 和 best MAE model 的残差
- reward 曲线
- MAE / best MAE 曲线

每个 run 的 `shot_wiggle_overlays.png` 包含：

- 每炮 shot gather 的 wiggle 对比
- black = true / observed reference
- red = initial、best、final 三种候选模型对应的合成炮集

因此这组可视化可以同时回答两个问题：

1. 速度模型是否接近真值？
2. 炮集事件是否在时间轴上对齐？

## 3. 总览结果

下面的柱状图是 8 类单 reward 在 10 个 CVA 模型上的 mean best MAE。数值越低越好。

[[FIG:summary_figures/reward_mean_best_mae.png|8 类 reward 的 mean best MAE 排名；error bar 是跨 10 个模型的标准差。]]

完整排名：

| Rank | Reward | Mean Best MAE | Median | Std | Mean Final MAE | Final-Best Gap | Wins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `wasserstein` | 61.90 | 49.43 | 42.30 | 95.34 | 33.44 | 2 |
| 2 | `wasserstein_w2` | 66.31 | 65.25 | 33.39 | 94.55 | 28.24 | 2 |
| 3 | `tt_only` | 71.41 | 70.41 | 40.77 | 105.07 | 33.66 | 2 |
| 4 | `awi` | 72.82 | 66.70 | 37.65 | 109.04 | 36.22 | 0 |
| 5 | `ncc_maxlag` | 73.24 | 67.26 | 36.98 | 117.80 | 44.56 | 0 |
| 6 | `envelope_ncc` | 74.42 | 68.95 | 37.71 | 109.12 | 34.70 | 0 |
| 7 | `ncc_zero` | 96.09 | 73.12 | 68.05 | 142.35 | 46.26 | 1 |
| 8 | `l1l2` | 108.89 | 64.32 | 94.25 | 148.11 | 39.21 | 3 |

关键观察：

- `wasserstein` 仍是最稳的单 reward。
- `wasserstein_w2` 的 final-best gap 最小，说明最后策略退化相对少。
- `tt_only` 修复 layout 后恢复了有效性。
- `ncc_maxlag` 没有崩，但没有成为最强单 reward。
- `ncc_zero` 和 `l1l2` 均值靠后，不过它们在个别模型上会赢。

## 4. 不同模型上的 reward 差异

[[FIG:summary_figures/reward_model_best_mae_heatmap.png|每个 reward 在每个 CVA 模型上的 best MAE heatmap。颜色越亮代表 MAE 越低。]]

逐模型最优 reward：

| CVA | Best Reward | Best MAE | Final MAE | Runner-up | Runner-up MAE |
|---:|---|---:|---:|---|---:|
| 1 | `wasserstein` | 21.44 | 41.19 | `ncc_maxlag` | 22.48 |
| 2 | `tt_only` | 82.74 | 108.41 | `wasserstein` | 83.22 |
| 5 | `wasserstein_w2` | 100.31 | 152.11 | `wasserstein` | 110.41 |
| 6 | `ncc_zero` | 31.58 | 32.36 | `l1l2` | 32.77 |
| 8 | `l1l2` | 21.27 | 23.84 | `wasserstein` | 47.61 |
| 10 | `wasserstein_w2` | 124.63 | 159.86 | `wasserstein` | 146.31 |
| 15 | `l1l2` | 13.34 | 14.37 | `wasserstein` | 22.81 |
| 16 | `l1l2` | 33.23 | 33.85 | `ncc_zero` | 56.35 |
| 18 | `wasserstein` | 4.53 | 6.43 | `wasserstein_w2` | 7.44 |
| 50 | `tt_only` | 36.86 | 57.26 | `l1l2` | 42.03 |

这个表说明：单 reward 的平均排名和逐模型最优并不完全一致。`l1l2` 平均最差，但在 CVA8/CVA15/CVA16 上反而最好；这意味着普通 waveform misfit 不是完全不可用，而是模型依赖性强。

## 5. Best model 与 Final model 必须分开汇报

[[FIG:summary_figures/reward_final_minus_best_gap.png|每个 reward 的 mean(final MAE - best MAE)。这个 gap 越小，说明策略最后越能留在好区域。]]

Phase6 一个非常重要的现象是：policy 经常能采样到一个很好的 best model，但 final model 会退化。

```math
\Delta_{\text{drift}}
= \operatorname{MAE}(v_{\text{final}}, v_{\text{true}})
- \operatorname{MAE}(v_{\text{best}}, v_{\text{true}})
```

因此后续所有汇报都应该同时报告：

- `best MAE`: reward 是否能把搜索带到好区域。
- `final MAE`: policy 是否真的稳定收敛。
- `final-best gap`: 训练后期是否漂移。

## 6. 代表性收敛曲线

[[FIG:summary_figures/selected_convergence_curves.png|代表性 run 的 current/display MAE 与 global best MAE 曲线。]]

从曲线看，很多 reward 的 `best_mae_global` 会在训练中显著下降，但 current/display MAE 会有回摆。这说明 reward 本身能提供搜索信号，但 PPO policy 稳定保持最优区域仍然是未解决的问题。

## 7. Cross Correlation 是否真的正确？

目前代码中的 Cross Correlation 有两个版本：

```math
R_{\text{NCC-zero}}
=
\operatorname{mean}_{s,r}
\frac{\langle d_{\text{pred}}^{s,r}, d_{\text{obs}}^{s,r} \rangle}
{\|d_{\text{pred}}^{s,r}\|_2 \|d_{\text{obs}}^{s,r}\|_2 + \epsilon}
```

```math
R_{\text{NCC-maxlag}}
=
\operatorname{mean}_{s,r}
\left(
\max_{\tau \in [-\tau_{\max},\tau_{\max}]}
\rho_{s,r}(\tau)
- \lambda |\tau^\*_{s,r}|
\right)
```

其中每条 trace 都是在时间轴上做归一化互相关。sanity check 结果如下：

| Case | `ncc_zero` | `ncc_maxlag` without penalty | `ncc_maxlag` with penalty |
|---|---:|---:|---:|
| same trace | 1.000 | 1.000 | 1.000 |
| amplitude ×3 | 1.000 | 1.000 | 1.000 |
| shift 12 samples | -0.170 | 1.000 | 0.940 |
| shift 40 samples | 0.000 | 1.000 | 0.800 |
| random noise | -0.020 | 0.154 | -0.067 |

所以现在的问题不是 “NCC 轴又错了” 或 “NCC 没被调用”，而是：

> 当前实现是 global full-trace NCC，不是导师文献里更强的 windowed time-lag cross-correlation。

global NCC 的弱点：

- 一条复杂 trace 中多个事件共享一个全局 lag，物理上太粗。
- 强事件会主导整条 trace 的相关峰。
- 多事件干涉时，max correlation 可能选到旁瓣或错误事件。
- 没有 confidence weighting，低质量窗口和高质量窗口被平均对待。

因此如果要真正验证导师说的 “Cross Correlation 应该稳定”，下一步应该做：

```math
R_{\text{windowed-TL}}
=
-\operatorname{mean}_{s,r,w}
q_{s,r,w}
\cdot
\operatorname{clip}
\left(
(\tau^\*_{s,r,w})^2, 0, \tau_{\max}^2
\right)
```

其中窗口质量权重可以取：

```math
q_{s,r,w} = \max(\rho_{s,r,w}^{\text{peak}},0)^\gamma
```

## 8. Reward 组别可视化证据

下面每个 reward 选一个代表性 run 展示。每组第一张图是四模型 + 残差 + 曲线，第二张图是每炮 wiggle overlay。

### 8.1 Wasserstein: CVA18

`wasserstein` 是 mean best MAE 第一名。CVA18 上 best MAE = 4.53，final MAE = 6.43，是这批实验最干净的成功样例。

[[FIG:runs/phase6_fixed_layout/FWI_wasserstein_cva18_seed42/phase6_visuals/models_residuals_curves.png|Wasserstein / CVA18：initial、true、best MAE、final model，以及残差和收敛曲线。]]

[[FIG:runs/phase6_fixed_layout/FWI_wasserstein_cva18_seed42/phase6_visuals/shot_wiggle_overlays.png|Wasserstein / CVA18：initial、best、final 的 shot gather wiggle overlay。]]

### 8.2 Wasserstein W2: CVA10

`wasserstein_w2` 在困难模型 CVA10 上表现最好，best MAE = 124.63，优于 `wasserstein` 的 146.31。

[[FIG:runs/phase6_fixed_layout/FWI_wasserstein_w2_cva10_seed42/phase6_visuals/models_residuals_curves.png|Wasserstein W2 / CVA10：困难模型上的最佳单 reward。]]

[[FIG:runs/phase6_fixed_layout/FWI_wasserstein_w2_cva10_seed42/phase6_visuals/shot_wiggle_overlays.png|Wasserstein W2 / CVA10：shot gather wiggle overlay。]]

### 8.3 TT-only: CVA50

`tt_only` 在 CVA50 上 best MAE = 36.86，是该模型最优。这说明修复 layout 后，走时 reward 重新有有效物理意义。

[[FIG:runs/phase6_fixed_layout/FWI_tt_only_cva50_seed42/phase6_visuals/models_residuals_curves.png|TT-only / CVA50：走时 reward 对背景速度的约束。]]

[[FIG:runs/phase6_fixed_layout/FWI_tt_only_cva50_seed42/phase6_visuals/shot_wiggle_overlays.png|TT-only / CVA50：shot gather wiggle overlay。]]

### 8.4 NCC maxlag: CVA1

`ncc_maxlag` 在 CVA1 上 best MAE = 22.48，接近该模型最优 `wasserstein` 的 21.44。它不是无效 reward，但目前没有在 10 个模型中拿到单模型第一。

[[FIG:runs/phase6_fixed_layout/FWI_ncc_maxlag_cva1_seed42/phase6_visuals/models_residuals_curves.png|NCC maxlag / CVA1：global cross-correlation 的代表性效果。]]

[[FIG:runs/phase6_fixed_layout/FWI_ncc_maxlag_cva1_seed42/phase6_visuals/shot_wiggle_overlays.png|NCC maxlag / CVA1：shot gather wiggle overlay。]]

### 8.5 NCC zero: CVA6

`ncc_zero` 平均表现偏弱，但在 CVA6 上拿到单模型第一，best MAE = 31.58。

[[FIG:runs/phase6_fixed_layout/FWI_ncc_zero_cva6_seed42/phase6_visuals/models_residuals_curves.png|NCC zero / CVA6：zero-lag NCC 的局部成功案例。]]

[[FIG:runs/phase6_fixed_layout/FWI_ncc_zero_cva6_seed42/phase6_visuals/shot_wiggle_overlays.png|NCC zero / CVA6：shot gather wiggle overlay。]]

### 8.6 Envelope NCC: CVA18

`envelope_ncc` 平均与 `awi`、`ncc_maxlag` 接近，但没有单模型胜出。它更适合作为 curriculum 的早期或辅助 reward。

[[FIG:runs/phase6_fixed_layout/FWI_envelope_ncc_cva18_seed42/phase6_visuals/models_residuals_curves.png|Envelope NCC / CVA18：包络相关的代表性结果。]]

[[FIG:runs/phase6_fixed_layout/FWI_envelope_ncc_cva18_seed42/phase6_visuals/shot_wiggle_overlays.png|Envelope NCC / CVA18：shot gather wiggle overlay。]]

### 8.7 AWI: CVA18

`awi` mean best MAE = 72.82，和 NCC 类方法接近。它也没有单模型胜出，但保留了 matching filter 的物理解释。

[[FIG:runs/phase6_fixed_layout/FWI_awi_cva18_seed42/phase6_visuals/models_residuals_curves.png|AWI / CVA18：matching-filter reward 的代表性结果。]]

[[FIG:runs/phase6_fixed_layout/FWI_awi_cva18_seed42/phase6_visuals/shot_wiggle_overlays.png|AWI / CVA18：shot gather wiggle overlay。]]

### 8.8 L1+L2: CVA15

`l1l2` 平均最差，但在 CVA15 上 best MAE = 13.34，final MAE = 14.37。这说明 waveform misfit 在简单/条件好的模型上仍然可以非常强。

[[FIG:runs/phase6_fixed_layout/FWI_l1l2_cva15_seed42/phase6_visuals/models_residuals_curves.png|L1+L2 / CVA15：传统 waveform misfit 的强成功案例。]]

[[FIG:runs/phase6_fixed_layout/FWI_l1l2_cva15_seed42/phase6_visuals/shot_wiggle_overlays.png|L1+L2 / CVA15：shot gather wiggle overlay。]]

## 9. 汇报时可以这样总结

本轮 Phase6 fixed-layout 结果可以这样对导师说：

1. 我们已经修复并验证了 trace-wise reward 的时间轴布局，当前 NCC/W1/W2/AWI/TT 都沿 `[shot, receiver, time]` 的最后一维计算。
2. 单 reward 排名里，Wasserstein 系列仍然最稳，`tt_only` 也恢复了有效性。
3. Cross Correlation 没有失败，但当前 global NCC/maxlag 不是最强；它需要升级到 windowed time-lag + confidence weighting 才更接近文献里稳定的 objective。
4. `best MAE` 与 `final MAE` 差距明显，说明 reward 能找到好模型，但 PPO policy 稳定保持好模型仍是问题。
5. 下一步最值得做的是 fixed-layout curriculum：`wasserstein -> contrastive/l1l2`，以及 `windowed_ncc_time_lag -> contrastive`。

## 10. 完整可视化索引

完整 80 组 run 的可视化索引文件为：

`phase6_visual_index.md`

其中每组都有：

- `models_residuals_curves.png`
- `shot_wiggle_overlays.png`

