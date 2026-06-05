# 实验清单 — VAE 潜空间 RL 地震反演

**更新**: 2026-05-18

---

## 一、VAE 预训练

| 实验 | 目录 | VAE | 训练数据 | Val MAE |
|------|------|-----|---------|--------|
| V1 | `vae_200ep` | 64D, β=1.0 | CVA 40文件 | 113.5 |
| V2 | `vae_200ep_128d` | 128D, β=1.0 | CVA 40文件 | 113.5 |
| V3 | `vae_joint_cva_fva` | 64D, β=1.0 | CVA+CVB 80文件 | 108.2 |
| V4 | `vae_joint_b01` | 64D, β=0.1 | CVA+CVB 80文件 | 59.4 |
| V5 | `vae_cvb_only` | 64D, β=1.0 | CVB 40文件 | 166.3 |
| V6 | `vae_fva_only` | 64D, β=1.0 | FVA 40文件 | 166.3 |

---

## 二、CVA 测试集潜空间 RL

| 实验 | 目录 | VAE | 步数 | CVA[50] | CVA[52] | CVA[55] | CVA[58] |
|------|------|-----|------|---------|---------|---------|---------|
| 200步基线 | `latent_200step_cva*` | V1 | 200 | 133 | 132 | 163 | 127 |
| 128D | `latent_128d_cva*` | V2 | 200 | 114 | 128 | 159 | 141 |
| 联合VAE | `latent_joint_cva50` | V3 | 200 | 124 | — | — | — |
| **5000步最佳** | `latent_5k_cva5*` | V3 | 5000 | **87.4** | **123.6** | — | — |
| β=0.1 5k | `latent_b01_cva50_5k` | V4 | 5000 | 54.9 | — | — | — |
| FWI-only | `latent_fwi_only_cva50` | V3 | 5000 | 89.5 | — | — | — |
| uf=2 | `latent_uf2_cva50_5k` | V3 | 5000 | 202 | — | — | — |
| CVB VAE→CVA | `latent_cvb_vae_cva50` | V5 | 5000 | 206 | — | — | — |

---

## 三、CVB 测试集潜空间 RL

| 实验 | 目录 | VAE | 步数 | CVB[0] | 备注 |
|------|------|-----|------|--------|------|
| CVA-only VAE | `latent_fva0/25/50` | V1 | 200 | 336 | 未训CVB |
| Joint VAE 200步 | `latent_joint_fva0` | V3 | 200 | 296 | |
| Joint VAE 5k | `latent_5k_fva0_v2` | V3 | 5000 | 288 | baseline |
| β=0.1 | `latent_b01_fva0_5k` | V4 | 5000 | 328 | |
| uf=2 | `latent_uf2_fva0_5k` | V3 | 5000 | 236 | 有帮助 |
| uf=3 | `latent_uf3_fva0` | V3 | 5000 | 318 | |
| β=0.1+uf=2 | `latent_b01_uf2_fva0` | V4 | 5000 | 308 | |
| CVB VAE→CVB | `latent_cvb_vae_cvb0` | V5 | 5000 | 303 | |
| G=64调参 | `cvb_G64_*` | V3 | 5000 | 324-328 | 无帮助 |

---

## 四、FVA 测试集潜空间 RL

| 实验 | 目录 | VAE | 步数 | FVA[0] |
|------|------|-----|------|--------|
| Joint VAE | `latent_fva_real_5k` | V3 | 5000 | 113.2 |
| CVB VAE→FVA | `latent_cvb_vae_fva0` | V5 | 5000 | 165.1 |

---

## 五、CMA-ES 对比

| 实验 | 目录 | 模型 | 代数 | MAE |
|------|------|------|------|-----|
| CVA[50] | `cmaes_cva50` | CVA[50] | 100 | 123.6 |
| CVA[52] | `cmaes_cva52` | CVA[52] | 100 | 145.6 |
| CVB[0] | `cmaes_fva0` | CVB[0] | 100 | 326.0 |

---

## 六、可视化状态

| 目录 | 状态 |
|------|------|
| 所有 `latent_5k_*` | ✅ 有 progression + summary |
| 所有 `latent_200step_*` | ✅ |
| 所有 `latent_128d_*` | ✅ |
| `cmaes_*` | ❌ 缺可视化 |
| `latent_fva_real_5k` | ✅ |
| `latent_fwi_only_cva50` | ✅ |
