# 走时 Reward 实现方案调研

**日期**: 2026-05-18 | **几何**: 透射式 (震源底部, 检波器地表)

---

## 1. 初至拾取方法

### 1.1 STA/LTA (最经典)

```
STA(t) = mean(|p[t-w_short:t]|)
LTA(t) = mean(|p[t-w_long:t]|)
ratio(t) = STA(t) / LTA(t)
初至 = ratio(t) 首次超过阈值的位置
```

- 优点: 简单、计算快、地震学标准
- 缺点: 阈值敏感、低 SNR 时差
- 参数: w_short=10, w_long=100, threshold=3.0

### 1.2 能量比 (更鲁棒)

```
E(t) = sum(p[t:t+w]²)
初至 = argmax(E(t+1)/E(t))
```

- 优点: 自适应、不需要调阈值
- 缺点: 需要窗口参数

### 1.3 振幅阈值 (最简)

```
threshold = α * max(|p|)  # e.g., α=0.05
初至 = 首次 |p[t]| > threshold 的位置
```

- 优点: 最简单
- 缺点: 噪声敏感

---

## 2. 透射几何下的走时 Reward

透射式采集：震源在底部 (z=nz-1)，检波器在地表 (z=0)。直达 P 波从底部传播到地表。

### 2.1 Reward 定义

```python
def traveltime_reward(p_pred, p_obs, dt=0.001):
    """
    p_pred, p_obs: [n_shots, n_receivers, nt] or [nt, n_receivers]
    Returns: scalar reward (higher is better, max=0)
    """
    # Extract first arrival times using energy ratio method
    t_pred = first_arrival_energy_ratio(p_pred, dt)
    t_obs  = first_arrival_energy_ratio(p_obs, dt)
    
    # Negative mean absolute time difference
    R_tt = -|t_pred - t_obs|.mean()
    return R_tt
```

### 2.2 实现细节

```python
def first_arrival_energy_ratio(p, dt=0.001, win=20):
    """
    p: [..., nt] — waveform at each receiver
    Returns: first arrival time (seconds) at each receiver
    """
    # Compute short-term energy
    energy = p.pow(2)
    
    # Sliding window energy ratio
    se = torch.zeros_like(energy)
    for t in range(win, len(energy)):
        e_short = energy[t-win:t].sum()
        e_long  = energy[max(0,t-5*win):t-win].sum() + 1e-10
        se[t] = e_short / e_long
    
    # First arrival = first time where energy ratio exceeds threshold
    threshold = 3.0 * se.mean()
    t_first = []
    for ch in range(se.shape[0]):
        idx = (se[ch] > threshold).nonzero(as_tuple=True)[0]
        t_first.append(idx[0].item() if len(idx) > 0 else len(se[ch]) - 1)
    
    return torch.tensor(t_first) * dt
```

### 2.3 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| win | 20 (0.02s) | 短窗口长度 |
| threshold | 3.0 × mean | 自适应阈值 |
| 最小到达时间 | 0.05s | 跳过近场噪声 |

---

## 3. 在 RL 框架中的使用

### 3.1 独立走时阶段

```python
# Step 1: 走时反演 (前 500 步)
R = R_tt  # 纯走时 reward
```

走时对**平均速度**敏感——如果模型整体偏快，直达波到得更早。走时能抓住长波长背景速度。

### 3.2 递进式

```python
# Step 1-500: 走时
R = R_tt
# Step 501-2000: 全波形
R = R_fwi
# Step 2001+: 多 reward
R = R_fwi + 0.05 * R_prior
```

### 3.3 混合 Reward (GDPO)

```python
A = w_tt * group_norm(R_tt) + w_fwi * group_norm(R_fwi)
# w_tt 从 1→0 退火，w_fwi 从 0→1
```

---

## 4. 预期效果

### 4.1 走时能抓住什么？

- **平均速度趋势**：整体偏快→早到→低 reward；整体偏慢→晚到→低 reward
- **大尺度结构**：厚低速层会导致明显的走时延迟
- **不敏感于细节**：小尺度的速度抖动对走时影响很小

### 4.2 走时不能抓什么？

- 精细层位边界
- 速度的小幅度波动
- 反射波信息

### 4.3 所以走时→FWI 递进是否合理？

**合理**。走时先抓长波长背景（防止 cycle-skipping），FWI 再精细化。这在传统 FWI 中叫"multi-scale inversion"——从低频到高频。在 RL 中，走时 reward 相当于"低频"信息。

---

## 5. 风险

1. **噪声敏感**：阈值需要根据实际数据调
2. **复杂模型失效**：强速度对比时，直达波可能被折射/散射，初至不是期望的路径
3. **计算开销**：每个 receiver 独立计算，但对 RL 来说微乎其微
4. **梯度信息**：走时 reward 对速度模型的梯度不是 everywhere 定义的（阈值导致）

> **结论**：能量比 + 自适应阈值的初至拾取方案在透射几何下是可行的，适合作为递进 reward 的第一阶段。
