# Phase 5B Design — Reward Combinations & Curriculum

**日期**: 2026-05-27  
**目标**: 基于 Phase IV + 5A 最佳单一 reward，测试混搭和递进策略

---

## 实现机制

两个可用机制，无需修改代码：

### A. Mixed（混合，GDPO 多 reward）
```bash
--fwi_type wasserstein --reward_l2_weight 1.0 --reward_tt_weight 1.0
```
GDPO 对各 reward 分量独立 group-normalize → 加权 → batch-normalize。天然支持 TT + 任一 fwi_type 的混合。

### B. Curriculum（递进，resume_ckpt）
```bash
# Stage 1: policy 在 reward A 下训练 2500 步
--fwi_type wasserstein --steps 2500 --out_dir runs/C1_stage1

# Stage 2: 从 stage1 最佳 policy 出发，换 reward B 再训 2500 步
--fwi_type contrastive --steps 2500 \
  --resume_ckpt runs/C1_stage1/policy_best.pt --out_dir runs/C1_stage2
```
`--resume_ckpt` 只加载 policy 权重，optimizer/config 全新。等价于"fine-tune with different reward"。

---

## 实验矩阵（10 组合 × 6 模型 = 60 runs）

### Group 1: Mixed（3 个，单 stage，5000 步）

| # | Reward | CLI |
|---|--------|-----|
| M1 | Wass + TT | `--fwi_type wasserstein --reward_l2_weight 1.0 --reward_tt_weight 1.0` |
| M2 | Contrastive + TT | `--fwi_type contrastive --reward_l2_weight 1.0 --reward_tt_weight 1.0` |
| M3 | NCC_maxlag + TT | `--fwi_type ncc_maxlag --reward_l2_weight 1.0 --reward_tt_weight 1.0` |

### Group 2: 2-Stage Curriculum（5 个，2500+2500 步）

| # | Stage 1 (2500) | Stage 2 (2500) | Rationale |
|---|---------------|----------------|-----------|
| C1 | Wasserstein_abs | Contrastive | Robust OT → waveform fine |
| C2 | Wasserstein_abs | L1+L2 | OT → amplitude detail |
| C3 | TT-only | Contrastive | Pure timing → waveform |
| C4 | NCC_maxlag | Contrastive | Lag penalty → waveform |
| C5 | Wasserstein_abs | NCC_maxlag | OT → explicit lag |

### Group 3: 3-Stage Curriculum（2 个，2000+1500+1500 步）

| # | Stage 1 (2000) | Stage 2 (1500) | Stage 3 (1500) |
|---|---------------|----------------|----------------|
| C6 | Wasserstein | Wass+TT (mix) | Contrastive |
| C7 | TT-only | Wass+TT (mix) | L1+L2 |

---

## 执行计划

| Round | GPU 1 | GPU 2 | GPU 3 | 依赖 |
|-------|-------|-------|-------|------|
| R1 | M1 (Wass+TT) | M2 (Contr+TT) | M3 (NCCm+TT) | 无 |
| R2 | C1 stage1 | C3 stage1 | C4 stage1 | 无 |
| R3 | C1 stage2 | C3 stage2 | C4 stage2 | R2 完成 |
| R4 | C2 stage1 | C5 stage1 | C6 stage1 | 无 |
| R5 | C2 stage2 | C5 stage2 | C6 stage2 | R4 完成 |
| R6 | C7 stage1 | — | — | 无 |
| R7 | C7 stage2 | C6 stage3 | C7 stage3 | R4/R5/R6 |

**总计**: ~7 rounds × 2-5h = 1-2 天。6 模型。

---

## 期望收益

Based on Phase IV + 5A results:

| Single Reward | Mean MAE | Strength |
|---------------|----------|----------|
| Wasserstein_abs | 67.0 | 最鲁棒，全模型不崩 |
| TT-only | 72.8 | 鲁棒，走时约束 |
| Contrastive | 75.0 | 困难模型最优 |

**假设**:
- Curriculum 应在简单模型上追近 Prog_TT→L2(4.1)，困难模型上追近 Contrastive
- Wass+TT mix 应比单一 Wass 更稳（TT 提供额外走时约束）
- 3-stage 应最接近"最优单模型 = 最佳组合"的理想

---

## 待定（取决于 Phase 5A 补充实验）

- [ ] Envelope_NCC v2 (fixed Hilbert) 结果 → 可能加入 curriculum
- [ ] Phase_func 结果 → 可能加入 C8: Phase_func → Contrastive
