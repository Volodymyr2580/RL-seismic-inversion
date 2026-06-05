#!/bin/bash
# Stage 1: TT-only 2500 steps (for C3 curriculum)
GPU=2
LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
PY=/data/shengwz/anaconda3/envs/devito/bin/python
cd /data/shengwz/swz/RL-seismic-inversion
for idx in 18 50 10 6 8 5; do
  echo "=== $(date) TT-only CVA[$idx] GPU=$GPU ==="
  $PY train_rl_fwi.py \
    --policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 \
    --cva_file_idx $idx --geometry transmission \
    --fwi_type l2 --reward_l2_weight 0.0 --reward_l1_weight 0.0 --reward_tt_weight 1.0 --reward_prior_weight 0.0 \
    --best_criterion l2 --steps 2500 --group_size 32 --ppo_epochs 4 --lr 5e-3 \
    --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 \
    --out_dir runs/curr_tt2500_cva${idx} --device cuda \
    --early_stop_patience 500 --early_stop_window 100 --save_every 100 --seed 42
done
echo "GPU2 ALL DONE (TT 2500)"
