# Phase 4 RL Seismic Inversion Reward 设计汇报

日期：2026-05-26

## 1. 汇报目的

导师给出的文献主要围绕一个核心问题：传统 FWI 的 L2 waveform misfit 太依赖逐点振幅匹配，容易在初始模型不准、低频不足、振幅不可信时出现 cycle skipping。因此，Phase 4 的 reward 设计不应只围绕 L1/L2 波形残差，而应更多引入 cross-correlation、time-lag、AWI 和 OT 这类更稳健的相位/走时/匹配型目标函数。

对当前 RL seismic inversion 来说，这个建议尤其重要。传统 FWI 需要从 objective 推导 adjoint source，而我们的 RL 框架只需要对每个候选速度模型给一个 reward。因此，很多传统 FWI 中实现很复杂的 non-differentiable 操作，例如 argmax time lag、窗口选择、置信权重，在 RL reward 中反而可以更直接地使用。

## 2. 先解释几个关键概念

### 2.1 Misfit 和 Reward 的关系

传统 FWI 通常写成最小化 objective/misfit：

$$
J(m)=\|d_{\text{pred}}(m)-d_{\text{obs}}\|^2
$$

这里 $J$ 越小越好。RL 中 reward 是越大越好，所以通常可以直接取负号：

$$
R(m)=-J(m)
$$

本报告里说“objective 更稳健”，对应到 RL 里就是“reward 对好模型的排序更可靠，不容易把 cycle-skipped 的坏模型打高分”。

### 2.2 Cycle Skipping

地震信号是振荡的。如果合成波形和观测波形差了超过半个周期，L2 可能会把合成波形错误地对齐到观测波形的另一个周期上。此时 loss 可能局部下降，但速度模型走向错误方向，这就是 cycle skipping。

简单说：L2 问的是“每个采样点的振幅是否一样”；但在早期模型较差时，更应该先问“事件大概是否在正确时间到达”。

### 2.3 Cross-correlation

Cross-correlation 可以理解为把一条 trace 沿时间滑动，寻找它和另一条 trace 最相似的位置：

$$
C(\tau)=\int d_{\text{obs}}(t+\tau)d_{\text{pred}}(t)\,dt
$$

最大值所在的 $\tau$ 就是 time lag，也就是两条 trace 的走时差：

$$
\Delta \tau = \arg\max_{\tau} C(\tau)
$$

如果 $\Delta \tau$ 接近 0，说明合成数据和观测数据在走时上对齐。这个思想比逐点 L2 更关注 phase/traveltime，因此更抗振幅错误。

### 2.4 Normalized Cross-correlation

Normalized cross-correlation 会除以两条 trace 的能量：

$$
\rho(\tau)=
\frac{\int d_{\text{obs}}(t+\tau)d_{\text{pred}}(t)\,dt}
{\sqrt{\int d_{\text{obs}}^2(t+\tau)dt}\sqrt{\int d_{\text{pred}}^2(t)dt}}
$$

这样如果两条 trace 只是整体振幅不同，相关系数仍然可以很高。因此它天然适合解决“真实数据和声学模拟的振幅不一致”问题。

### 2.5 Windowed Time-lag

如果整条 trace 里有直达波、反射波、多次波、散射波，直接在全 trace 上取一个 $\Delta \tau$ 会非常混乱。windowed time-lag 的做法是把观测数据分成多个时间窗，每个时间窗单独算 $\Delta \tau$ 和相关系数。

这也是导师提到的“对 dobs 开窗，分窗口计算 delta tau”。它思想简单，但实现 tricky，原因是窗口长度、窗口位置、频率阶段、噪声、缺失事件、多事件干涉都会影响 $\Delta \tau$ 的稳定性。

## 3. 文献脉络梳理

### 3.1 Cross-correlation 类 objective

#### Liu et al. 2017: normalized zero-lag cross-correlation

Liu et al. 提出 normalized zero-lag cross-correlation FWI，核心是最大化合成数据和观测数据在零时移处的归一化相关性。它相当于不再强迫逐点振幅相等，而是强调相位和形态相似。论文指出该方法对振幅误差、震源强度误差、非高斯噪声更稳健，但对 cycle skipping 仍然敏感，尤其当初始速度模型不够准时可能比 L2 更窄 basin of attraction。

对 RL reward 的启发：

- 可以作为一个非常低成本的 `R_ncc_zero`。
- 它适合替代当前部分 L2/L1 reward，降低振幅不匹配的影响。
- 但它不能单独解决大时移问题，所以不应作为唯一 reward。

参考：[Liu et al., 2017, GJI, DOI: 10.1093/gji/ggw485](https://doi.org/10.1093/gji/ggw485)。

#### Zhang et al. 2019: normalized nonzero-lag cross-correlation elastic FWI

Zhang et al. 将 cross-correlation 从 zero-lag 扩展到 nonzero-lag。这个变化很关键：zero-lag 只问“同一时间采样点是否相似”，nonzero-lag 则允许波形有一定时间偏移，并通过 lag 权重来鼓励最终对齐。它更接近 traveltime correction，也更适合初始模型不准时的 FWI。

对 RL reward 的启发：

- 当前 `contrastive reward` 已经用了最大归一化互相关，但更像全局 trace-level 相似度。
- 可以进一步把最大相关的 lag 显式纳入 reward：相关峰越高越好，lag 越接近 0 越好。
- 形式可以设计为：

$$
R_{\text{ncc-lag}}=\frac{1}{N}\sum_i \left[\max_{\tau\in[-\tau_{\max},\tau_{\max}]}\rho_i(\tau)-\lambda |\tau_i^*|\right]
$$

参考：[Zhang et al., 2019, Geophysics, DOI: 10.1190/geo2018-0082.1](https://doi.org/10.1190/geo2018-0082.1)。

#### Oh and Alkhalifah 2018: envelope-based global correlation norm

这篇文章把 envelope inversion 和 global correlation norm 结合起来。Envelope 能产生人工低频信息，帮助恢复长波长背景；global correlation norm 通过归一化相关降低振幅误差影响。论文采用两阶段策略：先用 envelope-based global correlation norm 建背景，再用普通 global correlation norm 提高分辨率。

对 RL reward 的启发：

- 当前 Phase IV 做过 envelope reward，但主要是 envelope L2，这仍然容易受 envelope 振幅尺度影响。
- 可以改成 envelope NCC：

$$
R_{\text{env-ncc}}=
\frac{\langle \text{env}(d_{\text{pred}}),\text{env}(d_{\text{obs}})\rangle}
{\|\text{env}(d_{\text{pred}})\|\|\text{env}(d_{\text{obs}})\|}
$$

- 这比 envelope L2 更符合文献思想。

参考：[Oh and Alkhalifah, 2018, GJI, DOI: 10.1093/gji/ggy031](https://doi.org/10.1093/gji/ggy031)。

### 3.2 Traveltime/time-lag 类 objective

#### Luo and Schuster 1991: wave-equation traveltime inversion

这是该方向的早期经典工作。它用波动方程正演合成 seismogram，再用 cross-correlation 提取合成和观测的 traveltime residual，然后最小化走时差：

$$
J_{\text{WTI}}=\frac{1}{2}\sum_i \Delta \tau_i^2
$$

优点是比 waveform L2 更线性，对速度背景更敏感，可以从较差初值恢复低/中波数速度结构。缺点是分辨率通常低于 FWI，而且需要能可靠识别和比较的相位事件。

导师指出的局限也在这里：早期 WTI 更适合初至波或窗口明确的事件；如果后续波形复杂，需要人工或自动开窗口，流程繁琐。

参考：[Luo and Schuster, 1991, Geophysics, DOI: 10.1190/1.1443081](https://doi.org/10.1190/1.1443081)。

#### Luo et al. 2016: full-traveltime inversion

Full-traveltime inversion 试图让反演“完全依赖走时信息”，减少振幅对 inversion 的劫持。它的思想是把 transmitted arrivals 和 reflected waves 的走时信息都纳入 wave-equation-based inversion，从而自动估计更可靠的运动学速度模型。

对 RL reward 的启发：

- 当前 TT-only reward 只基于初至拾取，信息太少。
- 下一步应该从“初至 TT”升级到“全波形 time-lag”，也就是不仅比较 first arrival，还比较后续反射/散射事件在局部窗口里的时移。

参考：[Luo et al., 2016, Geophysics, DOI: 10.1190/geo2015-0353.1](https://doi.org/10.1190/geo2015-0353.1)。

#### Zhang et al. 2018: CGG time-lag FWI

导师提到的 “CGG 内部使用 time lag objective” 对应这类工作。Zhang et al. 2018 用 frequency-dependent time windows 计算 traveltime misfit，并用 cross-correlation coefficient 作为权重，提升高质量走时测量的影响：

$$
\Delta \tau_w=\arg\max_{\tau}\int_{t_1}^{t_2}d_{\text{obs}}(t+\tau)d_{\text{pred}}(t)\,dt
$$

$$
J_{\text{TL}}=\sum_{s,r,w} c_w(\Delta\tau_w)\Delta\tau_w^2
$$

其中 $w$ 是窗口编号，$c_w$ 是归一化互相关系数。它的关键 trick 包括：

- 低频阶段窗口长一些，因为低频波let长、噪声相对强。
- 高频阶段窗口短一些，因为事件更清楚，时间测量可以更精细。
- 用相关系数给窗口加权，避免噪声窗口或匹配差的窗口主导更新。
- 主要用于降低振幅差异和 cycle skipping 的负面影响。

对 RL reward 的启发非常直接：我们可以定义 windowed time-lag reward：

$$
R_{\text{TL}}=-\frac{1}{N_w}\sum_{s,r,w} q_{s,r,w}\cdot \text{clip}(\Delta\tau_{s,r,w}^2,0,\tau_{\max}^2)
$$

其中 $q_{s,r,w}$ 可以取 $\max(c_w,0)$ 或 $c_w^\gamma$，低相关窗口可以直接降权。

参考：[Zhang et al., 2018, SEG Expanded Abstracts, DOI: 10.1190/segam2018-2997711.1](https://doi.org/10.1190/segam2018-2997711.1)。

### 3.3 AWI 类 objective

#### Warner and Guasch 2016: adaptive waveform inversion

AWI 的核心思想不是直接比 $d_{\text{pred}}$ 和 $d_{\text{obs}}$，而是先求一个 matching filter，使得一个波形卷积后可以变成另一个波形。如果速度模型正确，这个滤波器应该接近 zero-lag delta function。于是 AWI 的 objective 就是惩罚 matching filter 偏离零时移尖峰的程度。

直观解释：cross-correlation 是问“滑动多少时间最像”；AWI 是问“需要什么滤波器才能把合成数据变成观测数据”。如果这个滤波器的能量集中在零时移，模型就更可信。

对 RL reward 的启发：

- 可设计 `R_awi`，对每条 trace 求 Wiener matching filter $w(\tau)$，然后惩罚其时间扩散：

$$
R_{\text{AWI}}=-\frac{\sum_\tau |\tau|w^2(\tau)}
{\sum_\tau w^2(\tau)+\epsilon}
$$

- 由于 RL 不需要对 reward 可微，Wiener filter 的实现可以先用 FFT 版本，成本可控。
- AWI 比单纯 cross-correlation 更能处理波形差异，但实现和调参比 NCC/TL 更复杂。

参考：[Warner and Guasch, 2016, Geophysics, DOI: 10.1190/geo2015-0387.1](https://doi.org/10.1190/geo2015-0387.1)。

#### Yong et al. 2023: localized AWI

Localized AWI 指出传统 AWI 用一个全局 matching filter 处理整条 trace，但复杂地震信号里不同事件的 time shift 往往不同。一个全局 filter 会让多个事件互相干扰，导致 misfit 重新变得非凸。Localized AWI 用 Gabor/time-frequency 局部分析估计局部 matching filter，更适合复杂波形。

这和导师说的“分窗口算 delta tau 很 tricky”本质一致：复杂 trace 不能假设所有事件共享同一个时移。

对 RL reward 的启发：

- 先不急着完整实现 LAWI。
- 但可以采用它的简化思想：用多个局部窗口，而不是全 trace 单一 lag。
- 对窗口的置信度、宽度、重叠比例做消融实验。

参考：[Yong et al., 2023, GJI, DOI: 10.1093/gji/ggac496](https://doi.org/10.1093/gji/ggac496)。

### 3.4 OT 类 objective

本地 PDF 是 2026 年 Computational Geosciences 的文章，题为 “Multi-parameter full waveform inversion based on optimal transport distance in VTI media”。该文把 Kantorovich-Rubinstein norm 引入 VTI 多参数 FWI。它强调 OT 相比 L2 更能处理 seismic data 的 time shift，提高 objective convexity，缓解 cycle skipping。

需要注意：经典 OT 要求输入是正的概率分布且总质量相等，但地震波形有正负振荡且总能量不守恒。因此很多 OT-FWI 需要预处理，例如正负分离、平方、指数变换、graph-space transform 或 KR norm。

对当前项目的关键启发：

- 仓库当前 `wasserstein` reward 是对 trace 采样值排序后做 1D Wasserstein-like 距离。这个实现更像“振幅分布距离”，并不真正描述地震事件沿时间轴搬运的代价。
- 文献里的 OT-FWI 重点是 time-shift robustness。若要参考 OT，应考虑 KR norm、graph-space OT、或者至少用 time-axis 上的 cumulative distribution/soft-DTW/Sinkhorn-like 距离，而不是只排序振幅。
- 但是 OT 实现成本明显高于 TL/NCC，建议作为第二阶段方案。

参考：[Wei et al., 2026, Computational Geosciences, DOI: 10.1007/s10596-026-10427-4](https://doi.org/10.1007/s10596-026-10427-4)。

## 4. 与当前 Phase IV 结果的关系

当前已有 Phase IV 汇总显示：

- `TT-only` 很稳，说明走时信息确实能提供较可靠的大尺度约束。
- `FWI_Contrastive` 在难模型上表现好，说明相位无关或弱相位依赖的相似度有价值。
- `L1+L2` 在部分模型上效果好，但不是全局最稳。
- `FWI_Envelope`、`FWI_Windowed`、`FWI_Wasserstein` 从随机初值出发并没有稳定成功，说明这些 reward 的现有实现还没有充分体现文献中的关键设计。

因此，导师建议不是简单地“再加几个 reward 名字”，而是要把 reward 设计从 sample-wise amplitude residual 转向 event-wise phase/traveltime matching。

## 5. 推荐的 Phase 4 reward 改造方案

### 5.1 第一优先级：Windowed Time-Lag Reward

建议新增：

$$
R_{\text{TL}}=-\frac{1}{N}\sum_{s,r,w} q_{s,r,w}\cdot \Delta\tau_{s,r,w}^2
$$

实现要点：

- 对每条 trace 分窗口。
- 每个窗口内计算 normalized cross-correlation。
- 在有限 lag 范围内取最大相关峰位置作为 $\Delta\tau$。
- 用相关峰值作为质量权重 $q$。
- 对过大的 lag 做 clip，防止离群窗口毁掉 group reward。
- 先用固定窗口，再升级为频率相关窗口。

建议初始参数：

| 参数 | 建议 |
|---|---|
| lag 搜索范围 | `[-80, 80]` samples，后续按频率缩小 |
| 低频窗口长度 | `160-240` samples |
| 高频窗口长度 | `60-120` samples |
| 窗口重叠 | 50% |
| 低质量窗口阈值 | `corr_peak < 0.2` 降权或忽略 |
| reward 形式 | `-mean(q * delta_tau^2)` 或 `mean(q * exp(-delta_tau^2/sigma^2))` |

因为 RL 只需要打分，$\arg\max$ 不可微不是问题。

### 5.2 第二优先级：Envelope NCC 替代 Envelope L2

当前 envelope L2 的问题是仍然在比 envelope 振幅。建议改为：

$$
R_{\text{env-ncc}}=
\frac{1}{N}\sum_i
\frac{\langle e_{\text{pred},i},e_{\text{obs},i}\rangle}
{\|e_{\text{pred},i}\|\|e_{\text{obs},i}\|+\epsilon}
$$

它更接近 Oh and Alkhalifah 2018 的 envelope-based global correlation norm。

### 5.3 第三优先级：AWI-style Filter Reward

建议作为高级实验，不作为第一步主线。原因是 AWI 的 matching filter 会引入 water-level、频域稳定项、窗口长度等额外超参数。

可定义：

$$
w(\omega)=\frac{D_{\text{obs}}^*(\omega)D_{\text{pred}}(\omega)}
{D_{\text{obs}}^*(\omega)D_{\text{obs}}(\omega)+\epsilon}
$$

$$
R_{\text{AWI}}=
-\frac{\sum_\tau |\tau|w^2(\tau)}
{\sum_\tau w^2(\tau)+\epsilon}
$$

如果 windowed TL reward 已经明显有效，再考虑 AWI/Localized AWI。

### 5.4 OT reward 的重新定位

不建议继续把“排序后的 1D Wasserstein”作为主要 OT 方案，因为它会丢掉时间顺序，而 time shift 正是 OT-FWI 想解决的核心。

更合理的路线：

- 短期：保留现有 `wasserstein` 作为 baseline，但不要称为真正 OT-FWI。
- 中期：尝试 time-axis cumulative W1 或 soft-DTW。
- 长期：实现 KR norm 或 graph-space OT，但这会明显增加计算和调参成本。

## 6. 推荐实验路线

### Stage A: 快速验证 NCC/TL 是否比当前 reward 更稳

实验组：

1. `NCC_zero`: normalized zero-lag cross-correlation。
2. `NCC_maxlag`: 最大互相关加 lag penalty。
3. `TL_global`: 全 trace 单一 $\Delta\tau$。
4. `TL_windowed_fixed`: 固定窗口分段 $\Delta\tau$。
5. `TL_windowed_conf`: 固定窗口 + corr confidence weighting。

观察指标：

- 最终 MAE。
- best oracle MAE。
- reward 与 MAE 的 Spearman correlation。
- $\Delta\tau$ 分布是否逐步向 0 收敛。
- 低相关窗口比例是否过高。

### Stage B: Curriculum

推荐 curriculum：

| 阶段 | reward | 目的 |
|---|---|---|
| 0-30% steps | `TL_windowed_conf + prior` | 先抓长波长走时结构 |
| 30-70% steps | `0.7 TL + 0.3 env-ncc/contrastive` | 保持走时稳定，同时引入更多波形信息 |
| 70-100% steps | `0.3 TL + 0.4 contrastive + 0.3 L1/L2` | 细化结构和振幅/形态 |

在 GDPO 里，各 reward 先 group-normalize 再加权，这一点很适合多 reward 组合。

### Stage C: Advanced Reward

当 Stage A/B 有正结果后，再尝试：

- AWI filter spread reward。
- Local windowed AWI。
- KR/graph-space OT。

## 7. 可以向导师汇报的核心结论

1. 导师给出的文献可以分成四类：cross-correlation、traveltime/time-lag、AWI、OT。它们共同目标是降低 L2 waveform misfit 的非凸性，减少 cycle skipping 和振幅错误影响。

2. 对当前 RL seismic inversion，最值得优先实现的是 CGG time-lag objective 的 reward 化版本。因为 RL 只需要 reward，不需要 adjoint source，所以 windowed delta tau、argmax、置信权重都可以直接实现。

3. 当前项目已有 TT-only 和 Contrastive reward 的效果说明：走时/相关性信息确实比单纯 L2 更稳。但现有 TT 只用初至，信息不足；现有 contrastive 没有显式惩罚 lag；现有 Wasserstein 没有真正利用时间搬运结构。

4. 下一步建议把 reward 主线改成 `windowed time-lag + envelope NCC + contrastive/L1-L2 curriculum`，而不是继续堆叠普通 L2 或简单 envelope/Wasserstein。

5. 技术风险主要在窗口策略：窗口太长会混合多个事件，窗口太短会受噪声影响；低频应长窗，高频可短窗；低相关窗口必须降权。这正是工业界 time-lag objective 的 trick 所在。

## 8. 参考文献与链接

- Luo, Y., and Schuster, G. T. (1991). Wave-equation traveltime inversion. Geophysics, 56(5), 645-653. [DOI](https://doi.org/10.1190/1.1443081)
- Luo, Y., Ma, Y., Wu, Y., Liu, H., and Cao, L. (2016). Full-traveltime inversion. Geophysics, 81(5), R261-R274. [DOI](https://doi.org/10.1190/geo2015-0353.1)
- Liu, Y., Teng, J., Xu, T., Wang, Y., Liu, Q., and Badal, J. (2017). Robust time-domain full waveform inversion with normalized zero-lag cross-correlation objective function. GJI, 209(1), 106-122. [DOI](https://doi.org/10.1093/gji/ggw485)
- Oh, J.-W., and Alkhalifah, T. (2018). Full waveform inversion using envelope-based global correlation norm. GJI, 213(2), 815-823. [DOI](https://doi.org/10.1093/gji/ggy031)
- Zhang, Z., Alkhalifah, T., Wu, Z., Liu, Y., He, B., and Oh, J. (2019). Normalized nonzero-lag crosscorrelation elastic full-waveform inversion. Geophysics, 84(1), R15-R24. [DOI](https://doi.org/10.1190/geo2018-0082.1)
- Zhang, Z., Mei, J., Lin, F., Huang, R., and Wang, P. (2018). Correcting for salt misinterpretation with full-waveform inversion. SEG Expanded Abstracts, 1143-1147. [DOI](https://doi.org/10.1190/segam2018-2997711.1)
- Warner, M., and Guasch, L. (2016). Adaptive waveform inversion: Theory. Geophysics, 81(6), R429-R445. [DOI](https://doi.org/10.1190/geo2015-0387.1)
- Yong, P., Brossier, R., Métivier, L., and Virieux, J. (2023). Localized adaptive waveform inversion: theory and numerical verification. GJI, 233(2), 1055-1080. [DOI](https://doi.org/10.1093/gji/ggac496)
- Wei, S., He, B., Wang, M., and Lin, Y. (2026). Multi-parameter full waveform inversion based on optimal transport distance in VTI media. Computational Geosciences, 30, 27. [DOI](https://doi.org/10.1007/s10596-026-10427-4)
