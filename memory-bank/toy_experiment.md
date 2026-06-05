以下是为 Trae Solo Coder 准备的 `toy_experiment.md` 文件内容。你可以直接复制保存，作为编码指令。

```markdown
# Toy Experiment for GRPO Validation

## 1. 目的
设计一个**完全脱离地震正演的合成任务**，验证当前 GRPO 实现（策略网络 + map-level ratio + decoupled clipping）的正确性，并评估其在简单离散空间上的收敛速度。如果本实验顺利收敛，而实际 FWI/SI 实验不收敛，则问题大概率出在 reward 量级、正演保真度或 SI 计算；如果本实验也不收敛，则需优先排查 GRPO 代码自身的 bug。

## 2. 任务定义
- **目标**：策略网络学习输出一个**预先固定的隐藏控制点网格** `target_grid`（离散 bin 索引）。
- **状态空间**：无上下文——策略网络接收固定伪输入（全零张量），相当于学习一个**无条件策略**。
- **动作空间**：每个网格点独立从 `N_bins` 个离散类别中采样，整个网格为 `(H_ctrl, W_ctrl)` 个独立决策。
- **Reward**：采样网格与目标网格之间的 negative MSE（bin 索引级别）。
- **评估**：若算法正确，策略将逐步把概率质量集中到正确的 bin 上，reward 趋近于 0，采样的完整匹配准确率接近 100%。

## 3. 环境与组件
### 3.1 目标网格生成
```python
H_ctrl = 4          # 控制点高度
W_ctrl = 4          # 控制点宽度
N_bins = 10         # 离散类别数（故意很小，使搜索空间可控）
seed = 42

import torch
torch.manual_seed(seed)
target_grid = torch.randint(0, N_bins, (H_ctrl, W_ctrl))  # 4×4, 值 0~9
```
该目标网格在训练开始前生成一次，之后保持不变。

### 3.2 策略网络（复用现有模块）
- 使用项目中的 `SeismicViTControlPolicy`（或你已实现的任意策略网络类）。
- 输入：伪地震数据，形状 `(batch, N_s, N_r, N_t)`，内容全零。batch size = 1 即可（每个 step 采样 G 组动作）。
- 输出：logits 或 probs，维度 `(H_ctrl, W_ctrl, N_bins)`，经过 softmax 为概率分布。
- 注意：若现有策略网络强制要求输入特定维度，务必保证传入的伪数据形状匹配。

### 3.3 Reward 计算（新写，极简）
```python
def compute_reward(sampled_bins: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
    """
    sampled_bins: (G, H_ctrl, W_ctrl)  LongTensor，每组采样网格
    target_grid:  (H_ctrl, W_ctrl)     LongTensor
    返回: (G,)  tensor，每个元素的 reward
    """
    # 计算每个样本与目标的逐点平方误差，然后求平均
    diff = (sampled_bins.float() - target_grid.float().unsqueeze(0))  # (G, H, W)
    mse = (diff ** 2).view(sampled_bins.size(0), -1).mean(dim=1)   # (G,)
    reward = -mse
    return reward
```

### 3.4 GRPO 优化器（复用现有模块）
- 直接使用 `agents/grpo_optimizer.py` 中的 GRPO 类或函数。
- 确保 `compute_advantages` 和 `grpo_loss` 接受：`log_probs_old`, `log_probs_new`, `actions` (采样网格)，`rewards`，以及 `epsilon_low`, `epsilon_high`。
- 注意 actions 是采样的 bin 索引，形状 `(G, H_ctrl, W_ctrl)`。
- 现有实现可能按 map-level 计算比率：`m_i = exp(mean(log(π_new/π_old) ) over spatial positions)`。务必保持该语义。

### 3.5 训练循环
伪代码：
```
for step in range(max_steps):
    # 1. 策略网络前向（固定伪输入）
    logits = policy.forward(dummy_input)   # (1, H, W, N_bins)
    probs = F.softmax(logits, dim=-1)
    probs_old = probs.detach().clone()    # 用于后续 ratio 计算

    # 2. 采样 G 组动作
    actions_list = []
    log_probs_new_list = []
    for g in range(G):
        dist = Categorical(probs.squeeze(0))  # 假设每个位置独立
        action = dist.sample()                # (H, W)
        log_prob = dist.log_prob(action)      # (H, W) -> 之后在 GRPO 中取均值作为 map-level log prob
        actions_list.append(action)
        log_probs_new_list.append(log_prob)
    actions = torch.stack(actions_list)       # (G, H, W)
    log_probs_new = torch.stack(log_probs_new_list)  # (G, H, W)

    # 3. 计算 reward
    rewards = compute_reward(actions, target_grid)  # (G,)

    # 4. 计算 advantage（组内标准化）
    mean_r = rewards.mean()
    std_r = rewards.std()
    advantages = (rewards - mean_r) / (std_r + 1e-8)

    # 5. GRPO loss（需使用与采样时一致的 log_probs_old）
    with torch.no_grad():
        log_probs_old = log_probs_new.detach().clone()  # 采样时概率的 log
    loss = grpo_loss(log_probs_old, log_probs_new, advantages, actions,
                     epsilon_low, epsilon_high, ...)  # 根据你的实现调整参数

    # 6. 反向传播与更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 7. 记录指标
```

## 4. 超参数
| 参数 | 值 | 说明 |
|------|----|------|
| H_ctrl, W_ctrl | 4, 4 | 控制点网格 |
| N_bins | 10 | 极小的搜索空间 |
| G | 16 | 每 step 采样组数 |
| lr | 8e-4 | 学习率 |
| epsilon_low | 0.2 | GRPO clip 下界 |
| epsilon_high | 0.27 | GRPO clip 上界 |
| max_steps | 10000 | 总训练步数 |
| optimizer | Adam | 无 weight decay |
| dummy input shape | (1, N_s, N_r, N_t) | 值全零，N_s, N_r, N_t 可任意设，如 (1,1,1) 以最小化计算 |

## 5. 日志与评估指标
每个 step 后记录并定期打印（如每 50 step）：
- **Average Reward**：`rewards.mean().item()`
- **Perfect Match Accuracy**：采样中完全等于 `target_grid` 的比例 `(actions == target_grid).all(dim=(1,2)).float().mean()`
- **Position Accuracy**：每个位置预测正确的平均概率（需要从 probs 中取出目标 bin 的概率并平均）
- **Entropy**：策略熵，`- (probs * probs.log()).sum(dim=-1).mean().item()`，观察探索衰减

建议将以上指标打印到控制台并写入 TensorBoard 或 CSV。

## 6. 预期结果
若 GRPO 实现正确：
- Reward 应在前 1000 步快速上升（从约 -50~ -80 到接近 0）。
- Perfect Match Accuracy 应在 2000 步后稳定在 > 90%，5000 步后达到 100%。
- Position Accuracy 快速接近 1.0。
- Entropy 逐步下降但不会过早塌缩为 0（若塌缩过早且 reward 仍差，表明探索不足或 clipping 错误）。

## 7. 成功标准
- 在 10000 步内，最后 1000 步的平均 Perfect Match Accuracy ≥ 95%。
- 训练过程中 reward 曲线呈现平滑上升趋势，无明显剧烈震荡或 crash。

## 8. 交付物
编写一个独立的 **`toy_train.py`** 脚本，要求：
1. 导入项目中已有的 `agents`（至少包含策略网络和 GRPO 优化器）。
2. 遵循上述训练循环，无需读取任何外部数据文件。
3. 输出训练日志（控制台 + 可选 CSV）。
4. 添加随机种子固定（`torch.manual_seed(42)` 等）以确保可复现。
5. 脚本可以直接运行：`python toy_train.py`，无需额外命令行参数（或可接受简单参数如 `--steps`）。

## 9. 注意事项
- 若策略网络默认需要 DataLoader 或其他环境，请适配为直接接收 dummy tensor，不要引入多余依赖。
- GRPO 中 importance ratio 的计算要与采样时的分布一致，确保 `log_probs_old` 是采样时使用的确切概率（不是 forward 一次后再 eval 模式下的）。建议采样后立即 detach 保存。
- 确保所有张量在相同的设备上（CPU 或 GPU）。
- 梯度更新时不要意外修改 target_grid。
- 若代码中有任何与地震/正演相关的硬编码，请全部移除，本实验必须独立于 FWI 环境。
```

将此文档交给 Trae Solo Coder，它应当能据此生成一个简洁的验证脚本。如果你需要我直接提供完整的 `toy_train.py` 实现，也可以进一步补充。