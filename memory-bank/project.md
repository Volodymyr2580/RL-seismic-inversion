# project.md – Low-Dimensional RL-FWI with B-spline Parameterization

## 项目目标
本项目实现了一种 **基于强化学习（GRPO）的低维参数化全波形反演方法**。  
核心创新：
- 策略网络输出低维控制点（例如 `4×4`）上的速度离散分布。
- 通过 **B-spline 插值** 将控制点扩展为任意尺寸的速度模型（如 `70×70`）。
- 使用 **不可微正演模拟器**（`deepwave.scalar`）计算 reward，完全避免可微性要求。
- 支持三种奖励模式：纯 FWI 数据失配、纯 SI 能量（待补充）、混合奖励。

## 目录结构（建议）
project/
├── config/ # 配置文件
│ └── default.yaml
├── data/ # 数据集（软链接或存放路径）
│ ├── train/
│ └── test/
├── agents/ # 核心智能体组件
│ ├── policy_network.py # ViT 或 CBAM-UNet
│ ├── velocity_reconstructor.py # B-spline 插值
│ ├── reward_calculator.py
│ ├── grpo_optimizer.py
│ └── forward_simulator.py # 封装 deepwave
├── utils/ # 工具函数
│ ├── data_loader.py
│ ├── metrics.py # MAE, RMSE, SSIM
│ ├── visualization.py
│ └── logging.py
├── train.py # 主训练脚本
├── test.py # 测试与 TTO 脚本
├── AGENTS.md # 本文件同目录下
└── project.md # 本文件

数据格式
输入炮集：p 形状为 (N_s, N_r, N_t)，存储为 .npy 文件。

速度模型（仅用于测试评估，训练时不使用）：形状 (N_x_model, N_z_model)。
核心超参数（示例）
参数	默认值	说明
N_x_ctrl, N_z_ctrl	4, 4	控制点网格大小
N_bins	100	速度离散化 bin 数量
v_min, v_max	1500, 4500	速度范围（m/s）
N_groups	16	GRPO 每组采样个数
reward_mode	"fwi"	"fwi" / "si" / "mix"
lambda_mix	0.5	混合奖励权重（仅 mix 模式）
lr	8e-4	学习率
epsilon_low, epsilon_high	0.2, 0.27	GRPO clip 范围
log_transform_k, log_transform_c	3, 0	sign-preserving log 参数

实验记录与复现
使用git建仓库来维护程序版本，在progress.md文档中记录实现进展

关键随机种子固定（如 42）以确保可复现性。

训练过程中保存 checkpoint（每 N 步一次），包含策略网络状态和优化器状态。

当前状态与待办事项
基础架构设计（文档）

实现策略网络（ViT / CBAM-UNet）

实现 B-spline 插值模块

集成 deepwave 正演

实现 GRPO 优化器（参考 DeepWaveRL）

实现三种奖励模式（其中 SI energy 待定义）

完成训练主循环

在 OpenFWI 子集上验证


可参考程序：
E:\sci_research\GRPO-FWI\grpo_fwi\*
E:\sci_research\GRPO-FWI\train.py

CVA数据集保存位置 data\CVA\CurveVel_A\data （但我实验时只需要固定其中一个模型和对应的观测数据即可）

参考论文 22537_DeepWaveRL_Self_Supervis.pdf



### 重点说明SI的计算方式
可以参考 example_to_dyy\rtm_notes.ipynb
其中给出了在某一个model上计算RTM和计算RTM's SI的详细步骤

在具体的脚本中，可以充分利用example_to_dyy\funcs\draw_kernels.py和example_to_dyy\funcs\RTM.py两个函数


