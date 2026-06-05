# Phase IV Reward Functions Reference

本文档详述 Phase IV 所有实验中使用的 reward 计算方法，包括数学公式和实现细节。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $G$ | Group size（每组候选模型数，默认 32） |
| $N_s$ | 炮数（5） |
| $N_t$ | 时间采样点数（1000） |
| $N_r$ | 检波器数（70） |
| $\mathbf{p}_{\text{pred}}^{(g)} \in \mathbb{R}^{N_s \times N_t \times N_r}$ | 第 $g$ 个候选模型的正演地震记录 |
| $\mathbf{p}_{\text{obs}} \in \mathbb{R}^{N_s \times N_t \times N_r}$ | 观测地震记录 |
| $\mathbf{v}^{(g)} \in \mathbb{R}^{N_x \times N_z}$ | 第 $g$ 个候选速度模型 |
| $R^{(g)}$ | 第 $g$ 个样本的标量 reward（越大越好） |

---

## 1. Travel-Time Reward（TT）

### 1.1 初至拾取：能量比法

对每条 trace $\mathbf{x} \in \mathbb{R}^{N_t}$：

$$e[t] = x[t]^2$$

短窗能量和长窗能量之比：

$$\text{ratio}[t] = \frac{\sum_{i=t-W}^{t} e[i]}{\sum_{i=t-L-W}^{t-W} e[i] + \varepsilon}, \quad W=20, \; L=5W=100$$

初至时间：第一个 $\text{ratio}[t] > 3 \cdot \overline{\text{ratio}}$ 且 $t \geq 50$ 的采样点：

$$t_{\text{first}} = \min \{ t \mid \text{ratio}[t] > 3 \cdot \overline{\text{ratio}},\; t \geq 50 \}$$

### 1.2 走时差 Reward

对所有 trace 取平均初至时间差的负值：

$$R_{\text{tt}}^{(g)} = -\frac{1}{N_s N_r} \sum_{i=1}^{N_s N_r} \left| t_{\text{first}}(\mathbf{p}_{\text{pred},i}^{(g)}) - t_{\text{first}}(\mathbf{p}_{\text{obs},i}) \right|$$

### 1.3 Log 变换（`--reward_tt_log`）

放大低误差区间的区分度：

$$R_{\text{tt-log}}^{(g)} = -\log\left(-R_{\text{tt}}^{(g)} + 10^{-4}\right)$$

当 $|\overline{\Delta t}| = 1\text{ms} \to 100\text{ms}$ 时，$R_{\text{tt-log}} \in [4.6, 9.2]$。

---

## 2. Standard Data Misfit Rewards

### 2.1 Sign-Preserving Log Transform

先对地震数据做幅度压缩（$k=3.0, c=0$）：

$$\tilde{p}[t] = \text{sign}(p[t]) \cdot \log(k \cdot |p[t]| + c + 10^{-6})$$

### 2.2 L2 Misfit

$$R_{\text{l2}}^{(g)} = -\sum_{i=1}^{N_s} \sum_{t=1}^{N_t} \sum_{j=1}^{N_r} \left( \tilde{p}_{\text{pred},i,t,j}^{(g)} - \tilde{p}_{\text{obs},i,t,j} \right)^2$$

### 2.3 L1 Misfit

$$R_{\text{l1}}^{(g)} = -\sum_{i=1}^{N_s} \sum_{t=1}^{N_t} \sum_{j=1}^{N_r} \left| \tilde{p}_{\text{pred},i,t,j}^{(g)} - \tilde{p}_{\text{obs},i,t,j} \right|$$

---

## 3. Hilbert Envelope Misfit（`--fwi_type envelope`）

去掉相位信息，只保留瞬时振幅。Hilbert 包络通过 FFT 计算：

$$\mathcal{H}[x](t) = \mathcal{F}^{-1}\left[ \mathcal{F}[x](\omega) \cdot H(\omega) \right]$$

其中 $H(\omega) = 1_{\omega=0} + 2 \cdot 1_{\omega>0} + 1_{\omega=\pi} \cdot 1_{N_t\text{ even}}$。

$$\text{env}[x](t) = |\mathcal{H}[x](t)|$$

$$R_{\text{env}}^{(g)} = -\sum_{i,s,t,r} \left( \text{env}[p_{\text{pred}}^{(g)}] - \text{env}[p_{\text{obs}}] \right)_{i,s,t,r}^2$$

**优点**：抗 cycle skip，只关心"能量何时到达"而非"波形是否对齐"。

---

## 4. Time-Windowed L2（`--fwi_type windowed_l2`）

用 TT picker 确定初至窗口，只在该窗口内算 L2：

$$R_{\text{win}}^{(g)} = -\sum_{\text{traces}} \sum_{t = t_{\text{first}} - 30}^{t_{\text{first}} + 150} \left( p_{\text{pred},t}^{(g)} - p_{\text{obs},t} \right)^2$$

**缺点**：窗口外的散射波信息完全丢失。

---

## 5. 1D Wasserstein-1 Misfit（`--fwi_type wasserstein`）

对每条 trace，计算排序后样本的一阶 Wasserstein 距离（等价于 L1 距离）：

$$W_1(\mathbf{u}, \mathbf{v}) = \frac{1}{N_t} \sum_{k=1}^{N_t} \left| u_{(k)} - v_{(k)} \right|$$

其中 $u_{(k)}$ 是排序后的第 $k$ 个元素。对所有 trace 求和：

$$R_{\text{w}}^{(g)} = -\sum_{i=1}^{N_s N_r} W_1\!\left( \mathbf{p}_{\text{pred},i}^{(g)}, \mathbf{p}_{\text{obs},i} \right)$$

**优点**：比 L2 更凸，对相位错位不敏感。**缺点**：$O(N_t \log N_t)$ 排序开销。

---

## 6. Contrastive Reward（`--fwi_type contrastive`）

组合两个相位无关的相似度度量：

### 6.1 频谱余弦相似度

$$S_{\text{spec}}(\mathbf{x}, \mathbf{y}) = \frac{|\mathcal{F}[\mathbf{x}]| \cdot |\mathcal{F}[\mathbf{y}]|}{\| |\mathcal{F}[\mathbf{x}]| \| \cdot \| |\mathcal{F}[\mathbf{y}]| \|}$$

### 6.2 归一化互相关最大值

$$\bar{\mathbf{x}} = \frac{\mathbf{x} - \mu_x}{\sigma_x}, \quad \bar{\mathbf{y}} = \frac{\mathbf{y} - \mu_y}{\sigma_y}$$

$$\text{CC}_{\text{max}}(\mathbf{x}, \mathbf{y}) = \max_{\tau} \left[ \frac{1}{N_t} \sum_{t} \bar{x}[t+\tau] \cdot \bar{y}[t] \right]$$

### 6.3 组合 Reward

$$R_{\text{contra}}^{(g)} = \frac{1}{2N_s N_r} \sum_{\text{traces}} \left( S_{\text{spec}} + \text{CC}_{\text{max}} \right)$$

$R_{\text{contra}} \in [0, 1]$，值越大越相似。

---

## 7. Prior Reward

对速度模型施加地球物理先验约束：

$$R_{\text{prior}}^{(g)} = -\left( \omega_s \|\nabla_x \mathbf{v}^{(g)}\|_2^2 + \omega_s \|\nabla_z \mathbf{v}^{(g)}\|_2^2 + \omega_m M(\mathbf{v}^{(g)}) + \omega_b B(\mathbf{v}^{(g)}) \right)$$

其中：
- $\nabla$：空间梯度（平滑性，$\omega_s=1.0$）
- $M$：速度随深度递增约束（$\omega_m=0.1$）
- $B$：超出 $[v_{\min}, v_{\max}] = [1500, 4500]$ 的惩罚（$\omega_b=1.0$）

Phase IV 实验中 `--reward_prior_weight 0.0`（未使用）。

---

## 8. Multi-Reward via GDPO

### 8.1 多 Reward 组合

总 reward 为各分量的加权和，在 GDPO advantage 中实现：

$$R_{\text{total}}^{(g)} = w_{\text{l1}} R_{\text{l1}}^{(g)} + w_{\text{l2}} R_{\text{l2}}^{(g)} + w_{\text{tt}} R_{\text{tt}}^{(g)} + w_{\text{prior}} R_{\text{prior}}^{(g)}$$

### 8.2 GDPO Advantage

对每个 reward 分量独立做 group-standardize（在 $G$ 个样本内 z-score），再加权求和：

$$A_i^{(g)} = \frac{R_i^{(g)} - \mu_i}{\sigma_i}, \quad \mu_i = \frac{1}{G}\sum_g R_i^{(g)}, \quad \sigma_i = \sqrt{\frac{1}{G}\sum_g (R_i^{(g)} - \mu_i)^2}$$

$$A^{(g)} = \sum_i w_i \cdot A_i^{(g)}$$

最后全局 batch-standardize：

$$A^{(g)} \leftarrow \frac{A^{(g)} - \bar{A}}{\sigma_A}$$

### 8.3 PPO Clipped Loss

使用 token-mean PPO（每个控制点独立计算 ratio）：

$$\text{ratio}_{ij} = \exp\left( \log \pi_{\theta}(c_{ij} | \mu, \sigma) - \log \pi_{\theta_{\text{old}}}(c_{ij} | \mu, \sigma) \right)$$

$$\mathcal{L} = \frac{1}{G \cdot H \cdot W} \sum_{g,i,j} -\min\left( \text{ratio}_{ij} \cdot A^{(g)},\; \text{clip}(\text{ratio}_{ij}, 0.8, 1.27) \cdot A^{(g)} \right)$$

---

## 9. 各实验 Reward 配置

| 实验 | $w_{\text{l1}}$ | $w_{\text{l2}}$ | $w_{\text{tt}}$ | TT Log | FWI Type |
|------|:---:|:---:|:---:|:---:|------|
| TT-only | 0 | 0 | 1.0 | ✓ | — |
| Prog TT→L2 | 0 | 1.0 | 0 | — | l2 |
| Multi TT+L2 | 0 | 0.5 | 0.5 | ✓ | l2 |
| L1+L2 | 0.5 | 0.5 | 0 | — | l2 |
| FWI L2 | 0 | 1.0 | 0 | — | l2 |
| FWI Envelope | 0 | 1.0 | 0 | — | envelope |
| FWI Windowed | 0 | 1.0 | 0 | — | windowed_l2 |
| FWI Wasserstein | 0 | 1.0 | 0 | — | wasserstein |
| FWI Contrastive | 0 | 1.0 | 0 | — | contrastive |
| Prog TT→Contra | 0 | 1.0 | 0 | — | contrastive |
| Band 0-5Hz TT | 0 | 0 | 1.0 | ✓ | — |
| Band 0-10Hz Contra | 0 | 1.0 | 0 | — | contrastive |
| Band Full Contra | 0 | 1.0 | 0 | — | contrastive |

---

## 10. 频段滤波（Band Curriculum）

对 reward 计算前的 $p_{\text{pred}}$ 和 $p_{\text{obs}}$ 做 FFT 低通滤波：

$$X(\omega) = \mathcal{F}[x], \quad \tilde{X}(\omega) = X(\omega) \cdot H_{\text{LP}}(\omega; f_c)$$

$$H_{\text{LP}}(\omega; f_c) = \text{clamp}\!\left( \frac{1.2 f_c - f}{0.2 f_c + \varepsilon}, 0, 1 \right)$$

$$x_{\text{filtered}} = \mathcal{F}^{-1}[\tilde{X}]$$

其中 $f_c$ 为 cutoff 频率。过渡带 $[f_c, 1.2f_c]$ 做线性 ramp。TT reward 始终用原始未滤波数据（需要精确初至）。
