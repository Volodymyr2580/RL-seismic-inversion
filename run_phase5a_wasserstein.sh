#!/bin/bash
# Phase 5A: Wasserstein (fixed CDF-based W1, abs) on 6 CVA models
# Launch on GPUs 1,2,3 — 2 models each
set -e

LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
PROJ=/data/shengwz/swz/RL-seismic-inversion

# GPU 1: CVA18, CVA50
tmux new-session -d -s ws_gpu1 "bash -c '
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=1
cd $PROJ
for idx in 18 50; do
  echo \"=== \$(date) Wasserstein CVA[\$idx] GPU=1 ===\"
  python train_rl_fwi.py \
    --policy_type gaussian \
    --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx \$idx \
    --geometry transmission \
    --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
    --fwi_type wasserstein --wasserstein_normalize abs \
    --best_criterion l2 \
    --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 \
    --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 \
    --entropy_bonus 0.02 \
    --out_dir runs/FWI_WassersteinABS_cva\${idx} \
    --device cuda \
    --early_stop_patience 500 --early_stop_window 100 \
    --save_every 100 --seed 42
  echo \"=== \$(date) Done Wasserstein CVA[\$idx] GPU=1 ===\"
done
echo \"GPU1 ALL DONE\"
'"

echo "Launched GPU 1 (CVA18, CVA50)"

# GPU 2: CVA10, CVA6
tmux new-session -d -s ws_gpu2 "bash -c '
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=2
cd $PROJ
for idx in 10 6; do
  echo \"=== \$(date) Wasserstein CVA[\$idx] GPU=2 ===\"
  python train_rl_fwi.py \
    --policy_type gaussian \
    --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx \$idx \
    --geometry transmission \
    --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
    --fwi_type wasserstein --wasserstein_normalize abs \
    --best_criterion l2 \
    --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 \
    --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 \
    --entropy_bonus 0.02 \
    --out_dir runs/FWI_WassersteinABS_cva\${idx} \
    --device cuda \
    --early_stop_patience 500 --early_stop_window 100 \
    --save_every 100 --seed 42
  echo \"=== \$(date) Done Wasserstein CVA[\$idx] GPU=2 ===\"
done
echo \"GPU2 ALL DONE\"
'"

echo "Launched GPU 2 (CVA10, CVA6)"

# GPU 3: CVA8, CVA5
tmux new-session -d -s ws_gpu3 "bash -c '
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=3
cd $PROJ
for idx in 8 5; do
  echo \"=== \$(date) Wasserstein CVA[\$idx] GPU=3 ===\"
  python train_rl_fwi.py \
    --policy_type gaussian \
    --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx \$idx \
    --geometry transmission \
    --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
    --fwi_type wasserstein --wasserstein_normalize abs \
    --best_criterion l2 \
    --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 \
    --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 \
    --entropy_bonus 0.02 \
    --out_dir runs/FWI_WassersteinABS_cva\${idx} \
    --device cuda \
    --early_stop_patience 500 --early_stop_window 100 \
    --save_every 100 --seed 42
  echo \"=== \$(date) Done Wasserstein CVA[\$idx] GPU=3 ===\"
done
echo \"GPU3 ALL DONE\"
'"

echo "Launched GPU 3 (CVA8, CVA5)"
echo ""
echo "=== All 3 GPUs launched ==="
echo "Monitor: tmux attach -t ws_gpu1 | ws_gpu2 | ws_gpu3"
