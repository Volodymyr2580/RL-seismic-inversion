# RL-seismic-inversion 阶段性报告：项目启动至 Phase6

日期：2026-06-08  
用途：晚间与导师讨论的阶段性梳理  
范围：仓库内 `memory-bank/`、`outputs/`、`reports/`、`archive/`、`CHANGELOG.md`、`plan.md`、关键 reward/launcher 代码与服务器 Phase6 运行日志。

> 说明：我扫描了项目中的 Markdown/TXT 文档清单，并重点阅读了阶段总结、实验设计、结果报告、reward 文献综述、实验清单和 Phase6 修复相关代码。大量 `args.txt` / `command.txt` 属于单次运行参数记录，本报告只在需要复现实验设置时引用其归纳结论。

---

## 1. 一句话结论

这个项目从一开始的 “能不能用 RL 做单观测 FWI” 逐步演化成了一个更清楚的研究问题：

> 在低维速度模型参数化下，RL 可以把不可微、非传统的地震物理目标函数直接当作 reward 使用；真正决定稳定性的不是单纯的 PPO 技巧，而是 **参数化空间、reward 对候选模型的排序能力、以及 curriculum 的阶段切换设计**。

目前最强的历史结论来自 Phase5：在文献启发的 robust reward 中，Wasserstein / NCC / AWI / Contrastive 等比普通 L2 更有潜力，且 **curriculum 明显优于同时混合 reward**。但是 Phase6 暴露并修复了一个非常关键的 trace layout 问题，因此所有 “沿时间轴逐 trace 计算” 的旧结论都需要通过 corrected Phase6 重新确认。

---

## 2. 项目主线如何形成

### 2.1 初始目标

项目最初的目标是做 case-specific 单观测地震反演：

- 给定一个观测炮集 `p_data`。
- 策略采样一组低维速度控制点。
- 通过 B-spline 重建为速度模型。
- 用 Deepwave 正演得到合成炮集。
- 根据物理 reward 更新策略。
- 真值速度只用于 synthetic benchmark 的评估，不进入训练 reward。

这条路线的研究价值在于：RL 不要求 reward 可微，所以 cross-correlation 的 `argmax lag`、first-arrival picking、AWI matching filter、Wasserstein/KR 类目标都可以直接用于打分，而不必推导 adjoint source。

### 2.2 技术路线的核心转变

早期设计参考 DeepWaveRL，尝试过离散 bin 策略、ViT、大网络泛化式反演。但后续实验表明，当前任务不是大规模 supervised inverse model，而更像低维黑箱优化：

- 离散 ViT 策略很快 mode collapse，entropy 接近 0。
- 单观测场景中大网络过参数化，反而不稳定。
- 32 参数连续策略虽然简单，但更容易看清优化行为。
- 4×4 B-spline 控制点成为主要研究空间。

因此，项目主线从 “训练一个大网络反演器” 转向：

> 用小参数策略在低维速度空间中探索，并重点研究不同物理 reward 如何影响搜索方向。

---

## 3. Phase I：可行性验证与关键 bug 修复

### 3.1 做过的尝试

这一阶段主要在验证 RL pipeline 是否真的能学：

| 尝试 | 想法 | 结果 |
|---|---|---|
| Toy DAPO / 2×2 控制点 | 先脱离 Deepwave 验证策略优化逻辑 | 加入 search enhancement 和 per-token advantage 后能收敛到全局最优 |
| 离散 ViT 策略 | 对标 DeepWaveRL，用分类分布选择速度 bin | 全部 mode collapse，基本放弃为主线 |
| 连续 Gaussian + sigmoid | 把控制点当连续动作 | 明显比离散策略稳定 |
| 反射 / 透射几何 | 测试采集几何是否是主要瓶颈 | 透射几何没有单独解决优化问题，但后续成为主实验设置 |
| SI reward | 用 RTM imaging energy 作为结构约束 | 在 `v_true` 有反射界面时相关性强；在平滑合成模型上区分度弱 |

### 3.2 关键 bug：old_policy 同步时机

早期发现一个严重 bug：每步开始就把 `old_policy` 同步为当前 `policy`，导致 `logp_old == logp_new`，importance ratio 恒为 1，PPO/GRPO loss 基本没有有效梯度。

影响：

- `RL_results0` 一批早期结果不能算有效学习。
- 指标中 `ratio_mean=1.0`、`ratio_std=0`、`loss≈0` 是明显证据。

修复：

- 改成标准 PPO 四阶段：old policy rollout → 多 epoch 更新 → 诊断 → 同步 old policy。
- 加入 `--ppo_epochs`，默认进行多次 policy update。

这个 bug 的意义很大：它把项目从 “实验不收敛，不知道为什么” 拉回到 “RL pipeline 可以真实学习，但仍有参数化和 reward 难题”。

---

## 4. Phase II：B-spline 低维参数化与 baseline 建立

### 4.1 为什么用 B-spline

B-spline 的作用是把速度反演从 70×70 的 4900 个像素压到少量控制点。这样做有三个目的：

1. 降低 RL 搜索维度。
2. 强制速度模型平滑，符合早期背景速度反演。
3. 让策略输出更容易解释。

当时主要设置为 4×4 控制点，即 16 个速度控制点，策略参数约 32 个。

### 4.2 数据和基础设施

Phase II 建立了比较完整的 CVA 数据和训练基础设施：

- CVA 数据集：约 60 文件 × 500 样本。
- 训练/验证/测试划分：0-39 / 40-43 / 44-59。
- 支持反射几何与透射几何。
- CNN encoder 预训练入口可用。
- `train_rl_fwi.py` 成为后续主入口。

### 4.3 主要实验结果

| 实验 | 结果 | 解释 |
|---|---|---|
| 4×4 B-spline 理论拟合下限 | 平滑 3 层约 59 MAE，Marmousi 约 142，CVA[50] 约 195 | B-spline 本身有表达力上限 |
| CNN 预训练 | test MAE 约 280，接近纯 RL | warm start 有用但没有显著提升 |
| 纯 RL CVA[50] | best MAE 约 285 | 可学，但距离 B-spline 下限仍远 |
| 可微全像素 FWI | MAE 约 141 | reward 信号本身不是完全坏的 |
| 全像素 RL | MAE 约 707 | 高维 RL 搜索非常低效 |

Phase II 的核心判断：

> 问题不是 “FWI reward 完全没信息”，而是 RL 在高维/不合适参数化下效率太低。低维 B-spline 可行，但 4×4 控制点仍有表达力和优化瓶颈。

### 4.4 Seed 敏感性

三组较长实验显示 seed 差异非常明显：

| Seed | Best MAE | Final MAE | 现象 |
|---|---:|---:|---|
| 42 | 445.4 | 1413.1 | 早期碰到好点，后续严重漂移 |
| 7 | 373.0 | 698.1 | 有改进但 final 退化 |
| 77 | 376.6 | 403.3 | 晚期真正收敛，final 接近 best |

这说明需要区分 `best MAE model` 和 `final model`，否则会把搜索过程中偶然采到的好模型误认为收敛结果。

---

## 5. Phase III：VAE 潜空间与替代优化路线

Phase III 尝试过另一条参数化路线：用 VAE 把速度模型压缩到 latent space，再在 latent space 中做 RL。

### 5.1 做过什么

| 方向 | 设置 | 结果 |
|---|---|---|
| CVA-only VAE | 64D / 128D latent | Val MAE 约 113 |
| CVA+CVB joint VAE | 64D | Val MAE 约 108 |
| β=0.1 joint VAE | 64D | Val MAE 约 59 |
| Latent RL on CVA[50] | 5000 steps | 最好可到约 54.9-87.4 |
| CMA-ES baseline | CVA[50]/[52]/CVB[0] | CVA[50] 约 123.6 |

### 5.2 得到的判断

VAE latent RL 在若干 CVA 模型上明显优于早期 4×4 B-spline RL，例如 CVA[50] 可以做到 87 左右甚至更低。这说明 “更强的速度先验/参数化空间” 可以显著改善结果。

但后续主线仍回到 B-spline + reward 设计，原因是：

- B-spline 更透明，更容易解释 reward 对速度结构的影响。
- VAE decoder 引入了额外先验和重建误差，容易让 reward 研究变得混杂。
- Phase4/5 的核心问题变成：在一个可控低维空间里，哪种物理 reward 更稳定？

这个阶段可作为论文里的替代参数化/消融线索，而不是当前 Phase6 的主实验线。

---

## 6. Phase IV：从普通 waveform misfit 转向 robust reward

### 6.1 为什么进入 reward 设计

导师提醒的核心问题是：传统 L2 waveform misfit 依赖逐点振幅匹配，当初始模型不准或低频不足时容易 cycle skipping。

因此 Phase IV 不再只问 “L1/L2 能不能降”，而是开始尝试更接近地震事件匹配的 reward：

- Traveltime / first arrival。
- Cross-correlation / time-lag。
- Envelope。
- Contrastive waveform similarity。
- Wasserstein / OT 类距离。

### 6.2 Phase IV 实验矩阵

主要实验为 9 种策略 × 6 个 CVA B-spline 平滑模型：

| Strategy | CVA18 | CVA50 | CVA10 | CVA6 | CVA8 | CVA5 |
|---|---:|---:|---:|---:|---:|---:|
| TT-only | 10.2 | 36.9 | 150.0 | 58.4 | 75.1 | 106.0 |
| Prog TT→L2 | 4.1 | 41.3 | 200.1 | 51.7 | 69.1 | 141.2 |
| Multi TT+L2 | 45.8 | 31.9 | 159.7 | 42.4 | 69.8 | 139.7 |
| L1+L2 | 83.1 | 39.2 | 247.3 | 33.5 | 22.8 | 212.6 |
| FWI L2 | 339.2 | 188.3 | 259.8 | 270.6 | 40.2 | 273.9 |
| FWI Envelope | 60.8 | 41.9 | 163.6 | 38.3 | 84.9 | 148.7 |
| FWI Windowed | 372.4 | 325.4 | 234.2 | 270.8 | 104.0 | 278.5 |
| FWI Wasserstein-old | 365.1 | 241.1 | 228.1 | 108.4 | 115.8 | 277.2 |
| FWI Contrastive | 38.3 | 27.6 | 97.5 | 65.8 | 130.3 | 90.3 |

### 6.3 Phase IV 判断

| 观察 | 含义 |
|---|---|
| TT-only 很稳 | 初至走时确实能提供长波长背景约束 |
| Prog TT→L2 在简单模型上最好 | 先走时、后 waveform 的阶段策略合理 |
| Contrastive 在困难模型上强 | 相位/频谱相似度能缓解部分 cycle skipping |
| L1+L2 在少数模型上很好 | 普通 misfit 不是无用，但模型依赖强 |
| Envelope / Windowed / Wasserstein-old 不稳定 | 仅换 reward 名字不够，实现细节决定是否真的沿时间事件匹配 |

Phase IV 最重要的结论不是哪个 reward 绝对赢，而是：

> 没有单一 reward 能统治所有模型；reward 的作用取决于模型难度和训练阶段。

---

## 7. Phase V：文献驱动的 robust reward 与 curriculum

### 7.1 Phase5A：新单一 reward

Phase5A 根据 Phase4 reward 设计文档，把 NCC、Envelope NCC、AWI、CDF-based Wasserstein 等实现为单 reward。

| Reward | 设计思想 | Phase5A mean MAE |
|---|---|---:|
| Wasserstein_abs | 沿时间轴转非负密度，算 CDF-based W1 | 67.0 |
| AWI | matching filter 应接近 zero-lag delta | 77.9 |
| NCC_maxlag | 最大相关 + lag penalty | 78.7 |
| Envelope_NCC | Hilbert 包络后做 NCC | 80.4 |
| NCC_zero | 零时移归一化互相关 | 115.3 |

当时的判断：

- 修复排序式 Wasserstein 后，W1 从旧实现 mean 222.6 改到 67.0。
- NCC_zero 单独不够，因为它不能处理大 time shift。
- NCC_maxlag / AWI 中等有效。
- Wasserstein_abs 是最均衡的单 reward。

### 7.2 Phase5B：mixed vs curriculum

Phase5B 测试两类组合：

- Mixed：多个 reward 同时进入 GDPO，独立 group-normalize 后加权。
- Curriculum：先用 reward A 训练，再从 `policy_best.pt` 继续用 reward B fine-tune。

180-run 小规模完整排名显示：

| Rank | Strategy | Type | Mean MAE |
|---:|---|---|---:|
| 1 | Wass→Contrastive | Curriculum | 50.8 |
| 2 | TT→Contrastive | Curriculum | 61.1 |
| 3 | NCCm→Contrastive | Curriculum | 63.8 |
| 4 | Wass single | Single | 66.9 |
| 5 | Wass→NCCm | Curriculum | 70.6 |

关键结论：

> Curriculum > Mixed > Single。  
> 同时优化多个 reward 容易折中；先用 robust reward 找到合理区域，再用更细的 waveform/contrastive reward 精修，效果更稳定。

### 7.3 Phase5 large-scale：1520 次验证

Phase5 large-scale 扩展到 19 策略 × 16 模型 × 5 seeds = 1520 valid runs。

Top 结果：

| Rank | Strategy | Mean MAE | Std |
|---:|---|---:|---:|
| 1 | C1 Wass→Contrastive | 53.7 | 30.8 |
| 2 | C2 Wass→L1+L2 | 56.2 | 39.7 |
| 3 | C4 NCCm→Contrastive | 60.4 | 28.3 |
| 4 | C3 TT→Contrastive | 60.5 | 28.2 |
| 5 | C5 Wass→NCCm | 61.6 | 36.4 |
| 6 | Wasserstein single | 61.9 | 36.1 |
| 11 | AWI single | 65.6 | 34.5 |
| 13 | NCC_maxlag single | 70.6 | 31.7 |
| 18 | NCC_zero single | 115.4 | 77.2 |
| 19 | Phase_func | 210.8 | 86.0 |

Phase5 的研究价值很强：它给出一个连贯 story：

1. L2 容易 cycle skip。
2. 文献启发的 event-wise / phase-wise reward 有实际收益。
3. Wasserstein/NCC/AWI 这类目标能作为 RL reward 直接使用。
4. Curriculum 比简单混合更稳。

---

## 8. Phase VI：重新审计 NCC / W1 / AWI 的 trace layout

### 8.1 为什么需要 Phase6

Phase6 原本目标是做更严格的单 reward、单 seed、10 个 CVA B-spline 模型测试，并生成更完整可视化：

- initial model
- true model
- best MAE model
- final converged model
- best vs true residual
- final vs best residual
- reward/MAE curve
- shot gather wiggle 红黑对比：initial / best / last

测试 reward：

- `l1+l2`
- `tt_only`
- `wasserstein` / `wasserstein_w2`
- `ncc_zero`
- `ncc_maxlag`
- `envelope_ncc`
- `awi`

### 8.2 发现的问题

Phase6 初跑后发现一个严重布局问题：

- Deepwave/forward 部分实际保存或传出的预测炮集可能是 `[shot, time, receiver]`。
- `fwi_rewards.py` 等 trace-wise reward 按 `[shot, receiver, time]` 解包。
- 结果就是：W1/W2/NCC/AWI 这些应该沿时间轴计算的 reward，实际可能沿 receiver 轴在算。
- `tt_only` 也因此会把 `nt` 看成 70，first-arrival picker 基本失效，出现全 0 或无区分度结果。

这不是小误差，而是 reward 物理含义被改掉了：

> 本来要比较每条 trace 随时间的事件到达；错误布局下变成比较某个时间片上不同 receiver 的横向分布。

因此，Phase6 初跑结果不能作为正式科学结论，只能作为暴露 bug 的 evidence。

### 8.3 已完成的修复

代码层面已经做了统一：

- 新增 `agents/seismic_layout.py`，统一 canonical layout：
  - 单模型炮集：`[shot, receiver, time]`
  - batch 炮集：`[G, shot, receiver, time]`
- forward wrapper 立刻把 Deepwave 输出规范化。
- `train_rl_fwi.py` 加入训练时 shape assertion。
- `fwi_rewards.py` / `traveltime_reward.py` 加入 receiver/time swap fail-fast 检查。
- `tests/test_reward_sanity.py` 覆盖：
  - NCC zero 对振幅缩放不敏感。
  - NCC zero 对时间平移敏感。
  - NCC maxlag 能找回已知 shift。
  - TT reward 对 shifted trace 非零。
  - receiver/time swap 会被拒绝。

服务器验证记录：

- `8 passed in 5.29s`
- Deepwave smoke：
  - `simulate shape (5, 70, 1000)`
  - `batch shape (1, 5, 70, 1000)`
  - `tt_raw [-0.00698...]`
  - `tt_log [4.950...]`
  - `ncc_zero [0.841...]`

### 8.4 当前 Phase6 corrected rerun 状态

已在服务器 `/data/shengwz/swz/RL-seismic-inversion` 启动 corrected dense rerun：

- commit：`e4356df feat(phase6): add dense fixed-layout launcher`
- 输出目录：`runs/phase6_fixed_layout`
- launcher：`run_phase6_dense_grid.sh`
- 实验规模：8 rewards × 10 CVA models = 80 tasks
- 并行：4 张 CUDA 卡 × 每卡 4 个 worker = 16 workers
- seed：42
- 模型：`1 2 5 6 8 10 15 16 18 50`

我查询服务器时看到：

- 4 张 GPU 仍在高负载运行。
- 已产生多份 `metrics.csv`。
- 日志中已有多条 `DONE`，例如 `l1l2`、`tt_only`、`wasserstein`、`wasserstein_w2`、`ncc_zero`、`ncc_maxlag`、`envelope_ncc`、`awi` 的部分模型已完成。
- 当时未在日志片段中看到 `Traceback` / `ValueError` / `RuntimeError`。

---

## 9. 需要非常谨慎汇报的点

### 9.1 Phase5 结论的证据等级

Phase5 的统计规模很大，研究设计也清晰，但由于 Phase6 后续发现 trace layout bug，对以下 reward 的旧结果需要重新审计：

- Wasserstein / Wasserstein W2
- NCC zero / NCC maxlag
- Envelope NCC
- AWI
- TT-only
- Windowed L2 / Contrastive 中涉及 trace-time 操作的部分

它们的历史排名仍然有价值，因为它们说明了当时 pipeline 下的相对行为；但如果同一 layout 问题贯穿了旧实验，就不能直接把数值作为最终论文结论。

更稳妥的说法是：

> Phase5 建立了 reward 设计和 curriculum 假设；Phase6 fixed-layout rerun 是对这些假设的物理正确性复核。

### 9.2 哪些结论比较稳

相对更稳的结论：

- 早期 PPO old-policy bug 确实存在且已修复。
- 离散 ViT 策略在单观测场景中不适合作主线。
- 低维参数化比全像素 RL 更可行。
- B-spline 有表达力上限，VAE latent 空间更强但解释性更复杂。
- curriculum 思想在实验设计上合理，也符合 FWI multi-scale / long-to-short wavelength 逻辑。
- `best MAE` 与 `final model` 必须分开汇报。
- canonical layout 必须统一为 `[shot, receiver, time]`，否则 trace-wise reward 没有物理意义。

需要 Phase6 重新确认的结论：

- Wasserstein 是否仍是最佳单 reward。
- NCC_maxlag 是否优于 NCC_zero。
- AWI / Envelope_NCC 是否真的稳定。
- TT-only 的真实强度和失败模式。
- Phase5 large-scale 的 exact ranking 是否保持。

---

## 10. 晚上和导师可以怎么讲

### 10.1 建议开场

可以这样概括：

> 我们从一个 RL-FWI 可行性项目推进到现在，主要发现是：单观测反演里，策略网络越大不一定越好；低维速度参数化让 RL 搜索可行，但最终稳定性强烈依赖 reward 是否真的在比较地震事件的时间结构。Phase5 显示 curriculum reward 很有潜力，不过 Phase6 发现 trace layout 曾经不统一，所以现在正在做 fixed-layout 的严格复核。

### 10.2 可以重点展示的科学问题

1. **为什么 Cross Correlation 应该稳定？**  
   因为 NCC 降低振幅尺度影响，maxlag 又允许一定走时偏移，理论上比 zero-lag L2 更不容易 cycle skip。

2. **为什么之前 NCC 结果可能不可信？**  
   因为 reward 以为最后一维是 time，但实际可能是 receiver。这样 NCC 不是沿 trace 时间序列算，而是在 receiver 方向算，物理意义错了。

3. **为什么 TT-only 也会异常？**  
   初至拾取必须沿时间轴找 arrival。如果把 receiver 轴当 time，picker 看到的 “时间长度” 只有 70，first-arrival 逻辑自然失效。

4. **为什么 curriculum 比 mixed 更合理？**  
   同时混合 reward 会让策略在多个目标之间折中；而 FWI 传统上也强调从长波长到短波长。先用 W1/TT/NCC lag 找背景，再用 Contrastive/L1+L2 精修，更符合物理反演过程。

5. **下一步最重要的不是再加 reward，而是验证 reward 的轴和物理含义。**  
   Phase6 fixed-layout 就是这一步。

### 10.3 建议问导师的问题

1. 如果 Phase6 fixed-layout 后 NCC_maxlag 仍不如预期，是否优先做 windowed time-lag，而不是全 trace NCC？
2. 对 CVA B-spline 平滑模型，导师是否认可用 `best MAE` 作为 oracle 分析，同时单独报告 `final model` 作为收敛性指标？
3. Wasserstein/W1 如果 fixed-layout 后仍强，是否可以作为 curriculum 的第一阶段主 reward？
4. AWI 是否值得继续做 localized/windowed 版本？还是先把 NCC/windowed time-lag 做扎实？
5. 论文主线应更偏 “RL 可接入不可微 robust reward” 还是 “curriculum reward for RL-FWI”？

---

## 11. 当前建议的下一步

短期：

1. 等待 Phase6 fixed-layout 80 个任务全部完成。
2. 汇总每个 reward 的：
   - best MAE
   - final MAE
   - reward curve
   - MAE curve
   - best/final residual
   - wiggle shot gather overlay
3. 对比 fixed-layout 与旧 Phase6/Phase5 排名，明确哪些结论保留、哪些被推翻。

中期：

1. 如果 NCC_maxlag 仍不稳定，实现 windowed time-lag + confidence weighting。
2. 如果 W1 仍稳定，用 W1→Contrastive / W1→L1+L2 做 fixed-layout curriculum 复跑。
3. 把 `best MAE` 与 `reward-selected final` 区分成两个评价体系。

长期：

1. 重新选择参数化空间：4×4 B-spline、6×6/8×8 B-spline、VAE latent 三者做公平对比。
2. 把 reward robustness 和 parameterization robustness 分开写，避免结论混在一起。
3. 如果要写论文，主贡献可以收敛为：
   - RL 允许不可微 robust FWI objectives 直接作为 reward。
   - GDPO/curriculum 能协调不同物理 reward。
   - 低维地质参数化让单观测 RL-FWI 变得可计算。
   - 严格的 seismic tensor layout contract 是 trace-wise reward 的必要条件。

---

## 12. 证据来源索引

核心文档：

- `memory-bank/progress.md`：Phase IV/VI 进展与修复记录。
- `outputs/experiment_summary.md`：早期算法演变、old_policy bug、离散/连续策略结果。
- `outputs/debug_and_research_report.md`：old_policy 同步 bug 的详细分析。
- `outputs/phase2_status_summary.md`：Phase II 数据、B-spline、预训练、RL baseline。
- `outputs/phase2_final_report.md`：Phase II 三 seed 大实验。
- `reports/experiment_inventory.md`：VAE latent RL 与 CMA-ES 实验清单。
- `reports/bspline_inventory.md`：B-spline 相关实验清单。
- `reports/traveltime_research.md`：TT reward 设计。
- `reports/phase4_reward_design_report.md`：NCC/time-lag/AWI/OT 文献驱动设计。
- `reports/phase4_final/report.md`：Phase IV 9×6 结果。
- `reports/prog_contra/report.md`、`reports/curriculum/report.md`、`reports/band_curriculum/report.md`：progressive/curriculum 探索。
- `reports/phase5_results.md`：Phase5A 单 reward 结果。
- `reports/phase5b_design.md`：mixed 与 curriculum 设计。
- `reports/phase5_final/phase5_complete_report.md`：Phase5 180-run 完整结果。
- `reports/phase5_final/phase5_large_final.md`：Phase5 1520-run 大规模结果。
- `reports/reward_functions.md`：Phase IV reward 数学定义，但其中历史符号使用 `[shot,time,receiver]`，正好说明 layout 合约曾经不统一。

关键代码/脚本：

- `agents/seismic_layout.py`：当前 canonical layout 合约。
- `agents/fwi_rewards.py`：trace-wise reward 当前实现。
- `agents/traveltime_reward.py`：TT reward 当前实现。
- `tests/test_reward_sanity.py`：NCC/TT/layout sanity tests。
- `run_phase6_dense_grid.sh`：Phase6 corrected dense rerun launcher。

