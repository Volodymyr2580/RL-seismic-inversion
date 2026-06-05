# Phase 5: Robust Reward Functions for RL-FWI

**启动**: 2026-05-26  
**状态**: ✅ 5A 完成 → 进入 5B

---

## 阶段概述

Phase 5 将 reward 设计从 sample-wise 振幅残差（L2）转向 event-wise 相位/走时匹配。基于导师文献综述，实现 4+1 个新 reward：

| Reward | 类型 | 文献 | 状态 |
|--------|------|------|------|
| Wasserstein_abs | CDF-based W₁ (time-domain) | Métivier et al. 2016 | 🔴 训练中 |
| NCC_zero | Normalized zero-lag CC | Liu et al. 2017 | 📋 就绪 |
| NCC_maxlag | Max NCC + lag penalty | Zhang et al. 2019 | 📋 就绪 |
| Envelope_NCC | Envelope + NCC | Oh & Alkhalifah 2018 | 📋 就绪 |
| AWI | Wiener matching filter spread | Warner & Guasch 2016 | 📋 就绪 |

---

## 关键修复

### Wasserstein Bug（2026-05-26）
- **Bug**: `wasserstein1d()` 用 `.sort()` 比较排序振幅 → 振幅空间 W₁，完全盲于时间偏移
- **Fix**: 重写为 CDF-based time-domain W₁
- **验证**: 时间偏移信号 ratio = 10¹²（旧实现 = 0）

---

## 当前运行

| GPU | 实验 | 模型 | 进度 |
|-----|------|------|------|
| 1 | Wasserstein_abs | CVA18 | step 623, best MAE=9.3 ✅ |
| 1 | Wasserstein_abs | CVA50 | 等待 |
| 2 | Wasserstein_abs | CVA10 | step 129, best MAE=97.3 |
| 3 | Wasserstein_abs | CVA8 | step 670, best MAE=64.5 |

---

## 待启动

- Wasserstein_abs 完成后，4 个新 reward 各占 1 GPU
- 启动命令: `bash run_phase5a_batch.sh <gpu> <ncc_zero|ncc_maxlag|envelope_ncc|awi>`

---

## 文件变更

| 文件 | 变更 |
|------|------|
| `agents/fwi_rewards.py` | 重写 Wasserstein + 新增 NCC_zero/NCC_maxlag/Envelope_NCC/AWI |
| `train_rl_fwi.py` | 新增 `--fwi_type` 选项 + `--ncc_lag_{max,penalty}` + `--awi_version` |
| `reports/phase5_design_report.md` | Phase 5 完整设计文档 |
