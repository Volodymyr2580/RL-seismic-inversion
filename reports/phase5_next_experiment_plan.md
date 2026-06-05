# Phase 5 后续实验与 Reward Design 补充计划

**日期**: 2026-06-03  
**项目**: RL-guided seismic inversion / Phase 5 reward function design  
**目的**: 对 Sherlock 已完成的 Phase 5 实验进行后续补充规划，明确哪些实验尚未严格覆盖最初文献驱动的 reward design 方向，并提出下一轮测试建议。

---

## 1. 总体判断

Sherlock 已经完成了一轮很有价值的大规模实验，尤其证明了一个重要现象：

```text
curriculum > mixed > single
```

并且当前结果显示：

```text
Wasserstein 是很好的 Stage 1 reward
Contrastive / NCC_maxlag 等更适合后期精修
```

但是，Sherlock 的实现更像是“把 reward 菜单快速扩展并大规模扫了一遍”，而不是严格按照最初设计报告中的路线逐步推进：

```text
文献驱动 reward
→ reward 单元验证
→ single reward 消融
→ Top reward curriculum
→ 补导师重点方向
```

因此，下一轮最重要的工作不是继续盲目增加 reward 数量，而是补齐导师最关心、但目前尚未完整实现的 **windowed time-lag / localized event-wise alignment** 方向。

---

## 2. 当前主要缺口

### 2.1 最关键缺口：CGG-style windowed time-lag objective 尚未完整实现

导师最强调的是类似 CGG 的 `time lag objective`：

```text
对 dobs 分窗口
→ 每个窗口单独用 cross-correlation 估计 delta tau
→ 用相关系数作为置信度加权
→ 聚合为 reward / objective
```

当前 Phase 5 已经实现了：

| 已实现 reward | 说明 |
|---|---|
| `NCC_maxlag` | 在整条 trace 上搜索最佳 lag |
| `TT` | first-arrival picker，主要比较初至走时 |
| `+TT` mixed | 将 TT 作为辅助 reward 同时加入 |
| `TT→Contrastive` | 用 TT 作为 curriculum Stage 1 |

但这些仍然不是导师提到的 “开窗、分事件、分窗口算 delta tau” 的方法。

建议补充以下实验：

```text
TL_global
TL_windowed_fixed
TL_windowed_conf
TL_windowed_conf + multi-scale window
```

各实验含义：

| 实验 | 设计 |
|---|---|
| `TL_global` | 整条 trace 只计算一个 `delta tau`，作为最简单 time-lag baseline |
| `TL_windowed_fixed` | 固定窗口长度和 stride，对每个窗口独立计算 `delta tau` |
| `TL_windowed_conf` | 每个窗口计算 cross-correlation peak，低相关窗口自动降权 |
| `TL_windowed_conf + multi-scale` | 先用粗窗口稳定大走时，再用细窗口精修局部事件 |

建议 reward 形式：

```text
delta_tau_w = argmax_tau corr(pred_w, obs_w shifted by tau)
q_w = max(corr_peak_w, 0)^gamma
R_TL = - mean(q_w * clip(delta_tau_w^2, 0, tau_max^2))
```

其中：

| 参数 | 含义 |
|---|---|
| `delta_tau_w` | 第 `w` 个窗口的最佳时间偏移 |
| `q_w` | 该窗口的置信权重 |
| `gamma` | 控制低相关窗口被降权的强度 |
| `tau_max` | 最大允许 time lag，防止 reward 被异常大偏移主导 |

---

### 2.2 AWI 只做了简化版，尚未覆盖 AWI_full / Localized AWI

当前 Phase 5 的 AWI 实现是 matching filter 的 L1 spread：

```text
R_awi = - sum |tau| |w(tau)|^2 / sum |w(tau)|^2
```

这个版本是合理简化，但不是完整 AWI。最初设计中其实保留了两个层级：

| AWI 版本 | 状态 |
|---|---|
| `AWI_L1` | 当前已实现 |
| `AWI_full` | 尚未实现 |
| `Localized_AWI` | 尚未实现 |

建议补充：

```text
AWI_center
AWI_spread
AWI_full
Localized_AWI
```

含义如下：

| 实验 | 设计 |
|---|---|
| `AWI_center` | 只惩罚 matching filter 的中心偏移 |
| `AWI_spread` | 只惩罚 matching filter 的扩散宽度 |
| `AWI_full` | 同时惩罚中心偏移和扩散 |
| `Localized_AWI` | 对 trace 分窗口，每个窗口估计局部 matching filter |

建议公式：

```text
center = sum(tau * |w(tau)|^2) / sum(|w(tau)|^2)
spread = sum((tau - center)^2 * |w(tau)|^2) / sum(|w(tau)|^2)
R_AWI_full = - alpha * |center| - beta * spread
```

`Localized_AWI` 尤其值得做，因为它和导师强调的 windowed time-lag 思路高度一致：不要让整条 trace 的所有事件混在一个 matching filter 里，而是分事件、分窗口比较。

---

### 2.3 TT reward 目前只是 first-arrival surrogate，不是完整 FTI / WTI

当前 TT 更像工程化 picker：

```text
找到 pred 和 obs 的 first arrival
→ 比较 tau_pred - tau_obs
```

它符合 Luo & Schuster 1991 wave-equation traveltime inversion 的基础思想，但还没有覆盖后续波形事件，也没有实现完整 full traveltime inversion。

导师也指出了这一点：

```text
初至走时方法只能对初值波附近的波形有效。
如果有后续波形，需要另外开窗口，比较繁琐。
```

因此，TT 后续不建议继续只作为一个 `+TT weight=1.0` 的辅助项，而应升级成 windowed time-lag 的前置阶段。

建议补充：

```text
TT_first_arrival
TT_multi_window
TT_corr_weighted
TT_first → TL_windowed → Contrastive
```

---

### 2.4 Phase_func 没有直接文献支撑，应定位为 negative control

Sherlock 加入了 `Phase_func`，但它不是导师文献中明确提出的方法，也不是最初 reward design 的主线。

当前实验结果显示它表现最差，可以保留，但建议在论文或汇报中这样定位：

```text
Phase_func is a negative control showing that removing amplitude information destroys useful physical constraints.
```

也就是说，`Phase_func` 可以作为反例说明：

```text
完全去掉振幅信息会破坏 FWI 所需的物理约束。
```

不要把它包装成文献驱动 reward。

---

### 2.5 缺少 reward-level sanity check

最初设计报告里要求每个 reward 在训练前先做基本正确性验证。Sherlock 主要完成了大规模训练，但还缺少严格的 reward 单元测试。

这是很重要的，因为 RL 会非常认真地优化 reward。如果 reward 方向错了，训练结果也会系统性偏离。

建议新增测试文件：

```text
tests/test_reward_sanity.py
```

构造人工 trace：

```text
obs = Ricker wavelet
pred_same = obs
pred_shifted = shift(obs, +20)
pred_scaled = 3 * obs
pred_noisy = obs + noise
pred_wrong = random
```

每个 reward 应检查：

| 检查项 | 预期 |
|---|---|
| 相同信号 | reward 最大 |
| 时间平移 | reward 应下降，或显式识别正确 lag |
| 振幅缩放 | NCC 类 reward 应基本不变 |
| 随机信号 | reward 明显低于相同信号 |
| AWI zero-lag | matching filter 应最接近 delta |
| Windowed TL | 已知 shift 应被正确找回 |

---

### 2.6 缺少关键参数敏感性实验

当前 `NCC_maxlag` 使用：

```text
lag_max = 80
lambda = 0.005
```

但这些参数是否最优尚不清楚。`NCC_maxlag`、`AWI`、`windowed TL` 都对参数较敏感。

建议先做小规模 sweep，不要一开始就做 16 模型 × 5 seeds。

推荐代表模型：

| 难度 | 模型 |
|---|---|
| easy | `CVA1` / `CVA18` |
| medium | `CVA34` / `CVA43` |
| hard | `CVA10` / `CVA44` |

建议 sweep：

```text
NCC_maxlag:
  lag_max = 40, 80, 120
  lambda = 0.001, 0.005, 0.01

AWI:
  epsilon = 1e-4, 1e-3, 1e-2
  filter_length = short / full
  center_weight : spread_weight = 1:1, 2:1, 1:2

Windowed TL:
  window_length = 80, 120, 200
  stride = 40, 60, 100
  confidence_gamma = 1, 2, 4
  tau_max = 40, 80, 120
```

---

## 3. 下一轮实验优先级

### Priority 1: 补真正的 windowed time-lag

这是最贴合导师意见的方向，应作为下一轮最高优先级。

建议实验组：

```text
TL_global
TL_windowed_fixed
TL_windowed_conf
Wass→TL_windowed_conf
TL_windowed_conf→Contrastive
Wass→TL_windowed_conf→Contrastive
```

最推荐主线：

```text
Wass → TL_windowed_conf → Contrastive
```

设计原因：

| Stage | Reward | 作用 |
|---|---|---|
| Stage 1 | `Wass` | 先给粗结构，抗 cycle skipping |
| Stage 2 | `TL_windowed_conf` | 对齐多个波形事件，不只看初至 |
| Stage 3 | `Contrastive` | 后期做形态和频谱细修 |

---

### Priority 2: 升级 AWI

建议实验组：

```text
AWI_L1 当前版复查
AWI_center
AWI_spread
AWI_full
Localized_AWI
Wass→AWI_full
Wass→Localized_AWI
```

如果 `Localized_AWI` 表现好，它可以成为和 `TL_windowed_conf` 并列的主线方法。

---

### Priority 3: 补 reward sanity check 和参数 sweep

这一步不是为了“多跑几个训练”，而是为了证明 reward 本身确实在比较正确的物理量。

重点检查：

```text
NCC_maxlag 是否能正确找回已知 shift
AWI zero-lag 是否真的最优
Windowed TL 是否能忽略低相关窗口
Envelope_NCC 是否对振幅缩放不敏感
```

---

### Priority 4: 补论文叙事需要的对照实验

建议保留以下 baseline：

```text
L1+L2
TT_first_arrival
Contrastive
Wass
NCC_maxlag
AWI_L1
Phase_func negative control
```

然后与新方法对比：

```text
TL_windowed_conf
Localized_AWI
Wass→TL_windowed_conf→Contrastive
```

---

## 4. 建议的最小可执行实验矩阵

为了控制计算成本，建议先做一个小规模验证矩阵。

### 4.1 小规模 reward 验证

| 实验 | 模型 | Seed | Steps |
|---|---|---|---|
| `TL_global` | CVA1, CVA34, CVA10 | 42, 123 | 3000 |
| `TL_windowed_fixed` | CVA1, CVA34, CVA10 | 42, 123 | 3000 |
| `TL_windowed_conf` | CVA1, CVA34, CVA10 | 42, 123 | 3000 |
| `AWI_full` | CVA1, CVA34, CVA10 | 42, 123 | 3000 |
| `Localized_AWI` | CVA1, CVA34, CVA10 | 42, 123 | 3000 |

### 4.2 小规模 curriculum 验证

| 实验 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| C6 | Wass | TL_windowed_conf | — |
| C7 | TL_windowed_conf | Contrastive | — |
| C8 | Wass | TL_windowed_conf | Contrastive |
| C9 | Wass | Localized_AWI | — |
| C10 | Wass | Localized_AWI | Contrastive |

建议先用：

```text
3 models × 2 seeds
```

如果 C8 或 C10 明显优于当前最强结果，再扩大到：

```text
16 models × 5 seeds
```

---

## 5. 汇报中的建议表述

可以这样向导师汇报：

```text
Phase 5 已经验证了 curriculum reward scheduling 对 RL-FWI 明显有效，
并发现 Wasserstein 是较好的粗结构初始化 reward。

但当前实现尚未完全覆盖导师强调的 CGG-style windowed time-lag objective。
现有 NCC_maxlag 是 trace-level global lag matching，
TT 是 first-arrival surrogate，
二者都没有做到对 observed data 分窗口、逐事件计算 delta tau、并用 correlation confidence 加权。

因此下一步计划重点实现 TL_windowed_conf 和 Localized_AWI，
并测试 Wass → TL_windowed_conf → Contrastive 的三阶段 curriculum。
```

---

## 6. 一句话结论

Sherlock 已经证明了：

```text
curriculum 是有效的，Wasserstein 是很好的 Stage 1。
```

但尚未真正完成导师最关心的：

```text
windowed time-lag / localized event-wise alignment
```

下一轮最应该做的是把 `TL_windowed_conf` 和 `Localized_AWI` 做扎实，再用 sanity check 和参数 sweep 证明它们确实在比较“走时事件”，而不是只是在 reward hacking。

