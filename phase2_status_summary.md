## 2026-05-15 — Phase II 现状总结

### 已完成工作

**数据与基础设施**
- CVA 数据集：60 文件 × 500 样本 = 30,000 对 (p_data, v_model)
- Train/Val/Test: 0-39 / 40-43 / 44-59
- 反射几何 + 透射几何 均可在 `train_rl_fwi.py` 中切换
- SI reward 管线可用（RTM 调用对齐 Phase I），但在合成模型上无区分度

**B-spline 参数化验证**
- `agents/Bspline.py` 为你的原始代码，确认无误
- 4×4 B-spline 对不同模型的理论拟合下限：
  - 平滑 3 层: MAE=59
  - Marmousi: MAE=142
  - CVA[50]: MAE=195
- 注意：B-spline inverse 不约束 [1500,4500]，控制点可能越界（为拟合更好）
- 可视化中 `.T` 转置 bug 已修复（数据管线本身正确，仅影响 imshow 显示）

**CNN 预训练**
- 22,000 对训练（文件 0-43），2,000 对验证（44-47）
- Early stopping @ epoch 152，best val loss=0.034
- CNN 在 Test 集（48-51）上 MAE=280，接近纯 RL 的 285
- 预训练 checkpoint: `runs/cva_pretrain_20k/model_best.pt`

**RL 算法验证**
- 32 参数 mean policy (μ+κ) + PPO/GDPO 为稳定主线
- 最优配置：G=8, ppo_epochs=2, lr=5e-3, prior=0.05
- ratio 日志可信（不再恒为 1.000），clip 为 0（更新温和）
- 合成 3 层模型：best MAE=365
- CVA[50]：pure RL MAE=285，pretrain+RL MAE=288（无增益）
- CVA[52]：pure RL MAE=276
- 所有模型存在 "step 137 触顶后漂移" 特征

**全像素对照实验**
- 梯度 FWI（4900 参数，可微）：MAE=141（证明 FWI reward 信号强）
- 全像素 RL（4900 参数，不可微）：MAE=707（参数太多梯度太稀）
- 结论：问题不在 reward，在参数化效率

### 关键结论

1. **4×4 B-spline 表达力足够**，RL 差距来自优化效率
2. **Two-stage (CNN→RL) 无增益**：CNN 直接预测 ≈ 纯 RL
3. **FWI reward 信号没问题**：可微梯度 FWI 能到 141
4. **RL 瓶颈**：16 控制点参数化 + REINFORCE 梯度稀释
5. **下一步方向**：6×6 或 8×8 B-spline 增加表达力

### 待解决问题
- velocity bounds 应从 [1500,4500] 放宽到 [800,5000]
- steady state 漂移（best 在 step 137，之后退化）
- seed 敏感性
- 预训练 CNN 如何更好地为 RL 提供 warm start
