# B-spline 实验清单

**阶段**: Phase I & II | **参数化**: 4×4 B-spline 控制点 → 70×70 速度模型

---

## 一、Phase I: 探索阶段 (已归档)

| 实验 | 说明 | 结果 |
|------|------|------|
| 离散 ViT 策略 | DeepWaveRL 风格, 4组 sweep | 全部 mode collapse |
| 连续 32 参数 learnable | Gaussian + learnable params | MAE~34 (log_std 冻结) |
| old_policy 同步 bug | ratio≡1.0, PPO 未生效 | 已修复 |
| SI reward 测试 | RTM imaging energy | 仅在 v_true 上有区分度 |
| toy experiment | 脱离正演的离散 GRPO 验证 | 顺利收敛 |

> Phase I 详细记录: `archive/`, `outputs/experiment_summary.md`

---

## 二、Phase II: B-spline 基线

### 2.1 CVA 测试集 (PPO, G=32, ppo_epochs=4, 200步)

| 实验 | 目录 | CVA[50] | CVA[52] | CVA[55] | CVA[58] |
|------|------|---------|---------|---------|---------|
| B-spline 200步 | `bspline_200step_cva*` | 346 | 373 | 507 | 289 |

### 2.2 参数消融

| 实验 | 目录 | 配置 | CVA[50] |
|------|------|------|---------|
| Large G baseline | `mean_G32_cva50_50` | G=32, ppo=4, 50步 | ~346 |
| PPO epochs test | `mean_ppo8_synth` | G=8, ppo=8, synth model | ~490 |

### 2.3 CVA 预训练策略

| 实验 | 目录 | 说明 |
|------|------|------|
| Pure RL | `cva50_pure_rl` | 无预训练, 纯 B-spline RL |
| Pretrain + RL | `cva50_pretrain_rl` | CNN 预训练 warm start |
| Pretrain + RL noclip | `cva50_pretrain_rl_noclip` | 无 clipping 消融 |

### 2.4 Seed / Group size sweep

| 实验 | 说明 | 最佳 |
|------|------|------|
| `mean32_seed_group_sweep` | 15组 (5 seeds × 3 G) | G=8, seed=42 → MAE=364 |

---

## 三、关键发现

1. **B-spline 4×4 (16 控制点) 表达力封顶 ~195 MAE**（理论下限），RL 只能到 285-507
2. **32 参数 mean policy 是 Phase II 最稳定配置**
3. **CNN 预训练能 warm start 但全量微调不稳定**
4. **ratio≈1.0 根因**: ppo_epochs=2 太少 → PPO 来不及产生有意义的 ratio 偏差
5. **可微 FWI 做到 141** → 证明 reward 信号足够，瓶颈在 B-spline 参数化

---

## 四、B-spline vs VAE 总结

| 方法 | CVA[50] | CVA[52] | CVA[55] | CVA[58] |
|------|---------|---------|---------|---------|
| B-spline RL | 346 | 373 | 507 | 289 |
| VAE Latent RL | **87** | **124** | **159** | **127** |
| 改善 | ↓75% | ↓67% | ↓69% | ↓56% |

VAE 潜空间在所有模型上 2-4× 优于 B-spline。
