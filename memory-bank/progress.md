# Phase IV — Multi-Reward Seismic Inversion

**开始**: 2026-05-18 | **几何**: 透射式 | **模型**: CVA B-spline 平滑

---

## 实验记录

### 待完成

| # | 实验 | 状态 |
|---|------|------|
| 1 | 走时 reward 实现 | 📋 |
| 2 | 透射几何适配 (震源底部) | 📋 |
| 3 | B-spline 平滑模型生成 | 📋 |
| 4 | E1: 走时-only PPO (M1-M4) | 📋 |
| 5 | E2: FWI-only PPO (M1-M4) | 📋 |
| 6 | E3: 走时→FWI 递进 PPO (M1-M4) | 📋 |
| 7 | E4: 走时→FWI+Prior PPO (M1-M4) | 📋 |
| 8 | E5: FWI-only CMA-ES (M1) | 📋 |
| 9 | E6: 走时→FWI CMA-ES (M1) | 📋 |

---

## 已完成 Phase I-III

详见 `reports/` 文件夹。

---

## Phase VI — Corrected Single-Reward NCC Convergence Check

**启动准备**: 2026-06-06
**目的**: 在修复 reward 张量布局后，重新检查 Cross Correlation / NCC 类单 reward 在 CVA B-spline 重建模型上的 RL 收敛表现。

### 本轮关键修复

- 统一 `fwi_rewards.py` 的输入布局为 `[G, shot, receiver, time]` / `[shot, receiver, time]`。
- 修复受影响的 trace-wise reward: `windowed_l2`, `wasserstein`, `wasserstein_w2`, `contrastive`, `ncc_zero`, `ncc_maxlag`, `envelope_ncc`, `awi`, `phase_func`。
- 修复 `traveltime_reward.py` 的同类布局问题。
- 新增 `tests/test_reward_sanity.py`，用于验证 NCC 的振幅缩放不变性、时间平移敏感性、`NCC_maxlag` shift recovery 和 canonical layout。

### Phase6 默认实验设置

- Reward: `ncc_maxlag` 单 reward
- Seed: `42`
- Models: `1 2 5 6 8 10 15 16 18 50`
- 数据源: `data/smooth_models_v2` 中的 CVA B-spline 平滑模型
- 训练参数: 沿用 Phase5 单 reward 设置，`G=32`, `steps=5000`, `ppo_epochs=4`, `lr=5e-3`, transmission geometry
- 输出目录: `runs/phase6/FWI_ncc_maxlag_cva{idx}_seed42`

### 可视化要求

- 四个模型: initial, true, best MAE, final converged
- 残差: `|best - true|`, `|final - best|`
- 曲线: reward curve, MAE convergence
- 炮集: 每炮 wiggle 红黑对比，black=true, red=initial/best/final candidate

---

## Phase VII — Large-Contrast Model Sensitivity Setup

**启动**: 2026-06-11
**目的**: Phase6 讨论后，测试更复杂/更高敏感度速度模型在底部激发、表面接收透射观测下相对均匀模型的波形差异。

### 本轮新增

- 扩展 `agents/transmission_forward.py`，支持显式设置 `source_depth=bottom` 和 `receiver_depth=top`，默认 Phase6 行为保持不变。
- 新增 `scripts/phase7_model_sensitivity.py`：
  - 读取 `models/marmousi_v.bin`，按 `reshape(460,150)` 作为内部 `[nx,nz]`；绘图时再转置为 `[z,x]`。
  - 读取 `models/vel_z6.25m_x12.5m_exact.bin`，文件实际 float32 数量对应 `nz=1911, nx=5395`。
  - 支持 Salt 左上角裁剪、重采样到 `nx=210, nz=70`。
  - 对选定模型做 B-spline inverse/prolong 得到实验 `v_true`。
  - 使用底部炮点、表面接收点，对比 `v_true` 和均匀速度模型的归一化 wiggle 波形。

### 初始 sensitivity 结果（方向修正版）

| Model | Crop / Shape | Ctrl | rel L2 | RMSE | Corr | 输出 |
|---|---:|---:|---:|---:|---:|---|
| Marmousi | full -> `210x70` | `30x10` | 1.5868 | 0.1274 | 0.1509 | `outputs/phase7_sensitivity/marmousi_nx210_nz70_ctrl30x10` |
| Salt top-left | `x0=0,z0=0,nx=840,nz=280` -> `210x70` | `30x10` | 0.4084 | 0.0371 | 0.9149 | `outputs/phase7_sensitivity/salt_nx210_nz70_ctrl30x10` |
| Synthetic transmission-sensitive | procedural `210x70` | `30x10` | 1.5742 | 0.0917 | -0.0467 | `outputs/phase7_sensitivity/synthetic_nx210_nz70_ctrl30x10` |

### 方向修正说明

- 旧版 `ctrl10x30` 结果存在两个问题：Marmousi 内部数组被误当作 `[nx,nz]` 后绘图再次转置；B-spline 控制点顺序也被误写成纵向 10、横向 30，实际代码解释为 `[nx_ctrl,nz_ctrl]=[10,30]`。
- 修正后统一约定：保存/正演/B-spline 使用 `[nx,nz]`，绘图时用 `.T` 显示为 `[z,x]`。
- Salt 二进制文件应按 Fortran/MATLAB 顺序读取：`reshape((1911,5395), order="F")`；普通 C-order 会产生非地质的横条纹假模型。
- 当前建议 Salt crop A：`x0=0, z0=0, crop_nx=1800, crop_nz=1200`，输出检查图在 `outputs/phase7_sensitivity/salt_full_overview`。
- Salt crop A benchmark：
  - `target_nx=270,target_nz=90,ctrl=36x12,n_shots=7,n_receivers=270,nt=1000`，完整流程约 51 秒，`rel_l2=1.9438,corr=-0.0936`。
  - `target_nx=650,target_nz=250,ctrl=65x25,n_shots=7,n_receivers=650,nt=1000`，完整流程约 60 秒，`rmse=0.0729,corr=0.0220`；图像孔径太大且时间窗偏短，暂不作为主方案。
  - 后续 Salt 主方案暂定 `90x270`，并优先查看 `salt_A_90x270_global_compare.png` 这类全局缩放图，避免单道归一化 wiggle 掩盖振幅和多路径差异。
- 新增 `scripts/phase7_make_compare_panels.py`，统一为 Marmousi、Salt A、Synthetic 生成：
  - `model_annotated.png`：标注 `nx,nz,dx,dz`、底部炮点、表面接收点和正演参数。
  - `global_gather_compare.png`：`v_true`、均匀模型、差值和全局缩放 wiggle 对比。
  - 输出目录：`outputs/phase7_sensitivity/compare_panels`。

### 当前判断

- Marmousi 和合成透射敏感模型对当前观测模式的波形差异明显，可以进入下一轮控制点网格和 reward 测试。
- Salt 左上角默认框相对均匀模型仍较相似，后续需要根据蓝框位置继续调整裁剪区域，优先寻找包含明显盐体边界/大尺度横向速度异常的区域。

---

## Phase VII — Reward Suite 实验脚本准备

**更新**: 2026-06-14
**目的**: 在 Marmousi、Salt crop A `90x270`、Synthetic 三个模型上，系统测试 `tt_only`、`l1+l2`、`wasserstein(w1/w2)`、`ncc_zero`、`ncc_maxlag`、`envelope_ncc`、`awi` 奖励函数。

### 本轮新增

- 扩展 `train_rl_fwi.py`：
  - `--model_source npy --model_path ...`：直接读取 Phase7 已准备好的 `[nx,nz]` 速度模型。
  - `--init_velocity_path ...`：把确定性初始速度模型通过 `bspline2d_inverse` 投影到控制点，再初始化 `mean/gaussian` policy。
  - `--source_depth bottom --receiver_depth top`：训练入口正式支持 Phase7 的底部激发、表面接收几何。
  - 修正训练 summary/progression 图的显示方向：速度数组内部仍为 `[nx,nz]`，绘图统一 `.T` 为 `[z,x]`。
- 新增 `scripts/phase7_prepare_initial_models.py`：
  - 每个真实模型生成三个非随机初始模型：`far_uniform`、`medium_gradient`、`near_blur`。
  - 输出目录：`outputs/phase7_sensitivity/initial_models`。
- 新增 `scripts/phase7_visualize_run.py`：
  - 对单个 run 生成 `models.png`、`gather_images.png`、`gather_residuals.png`、`wiggle_overlay.png`、`metrics.png`。
- 新增 `scripts/phase7_build_report.py`：
  - 扫描 `runs/phase7`，生成 `reports/phase7_reward_suite/index.html`、`summary.csv`、`summary.json` 和 MAE heatmap。
  - 支持训练未完成时生成 partial report。
- 新增 `run_phase7_reward_suite.sh`：
  - 一键调度 72 组实验：`3 models × 3 initial models × 8 rewards`。
  - 默认 `GPU_LIST=0,1,2,3`，四卡并行；每张卡内部串行跑任务。
  - 支持 `MODELS`、`INITS`、`REWARDS` 环境变量筛选子集。

### 验证

- `python -m py_compile train_rl_fwi.py scripts/phase7_prepare_initial_models.py scripts/phase7_visualize_run.py scripts/phase7_build_report.py` 通过。
- 本地 `dw_env` 冒烟测试通过：
  - 使用 Synthetic `v_true.npy` + `medium_gradient.npy`，`n_shots=2,n_receivers=32,nt=200,G=2,steps=1`。
  - 成功完成 `npy` 读取、bottom-source/top-receiver 正演、初始模型注入、训练落盘和 per-run 可视化。
  - 冒烟测试的 wiggle 仅用于检查代码链路；由于 `nt=200` 时间窗很短，不作为物理结果解释。

### 服务器完整运行结果

**完成时间**: 2026-06-15
**服务器路径**: `/data/shengwz/swz/RL-seismic-inversion`
**运行规模**: `3 models × 3 initial models × 8 rewards = 72` 组实验，全部 `complete`。
**报告**: `reports/phase7_reward_suite/index.html`

每个模型的最佳组合：

| Model | Best init | Best reward | Best MAE (m/s) |
|---|---|---|---:|
| Marmousi | `near_blur` | `tt_only` | 207.74 |
| Salt A `90x270` | `near_blur` | `tt_only` | 242.70 |
| Synthetic | `near_blur` | `ncc_zero` | 192.71 |

按 reward 观察：

- `tt_only` 在 Marmousi 和 Salt A 上表现最好，Synthetic 上也接近最优。
- `ncc_zero` 在 Synthetic 上最好，在 Marmousi/Salt A 上也稳定排在前列。
- `wasserstein_w2` 在 Marmousi 上明显优于 `wasserstein_w1`，但在 Salt A/Synthetic 上不占优。
- `l1+l2`、`awi` 和 `wasserstein_w1` 在本轮设置下整体偏弱，部分组合触发 early stop（例如 Marmousi/Synthetic 的 `wasserstein_w1`）。
- 初始模型影响很强：`near_blur` 几乎支配所有模型和 reward 的最佳结果；`far_uniform` 明显困难，Salt A 的 far start 最优仍约 945 m/s MAE。
