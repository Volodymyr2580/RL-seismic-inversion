#!/bin/bash
# Multi-frequency curriculum: 5Hz TT → 10Hz Contra → 15Hz Contra
GPU=$1; IDX=$2
LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
cd /data/shengwz/swz/RL-seismic-inversion

BASE="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx $IDX --geometry transmission --reward_l1_weight 0.0 --reward_prior_weight 0.0 --best_criterion l2 --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100 --seed 42"

# Stage 1: TT-only at 5Hz
echo "=== $(date) Stage1: TT@5Hz CVA[$IDX] ==="
python train_rl_fwi.py $BASE --freq 5.0 --reward_tt_weight 1.0 --reward_tt_log --reward_l2_weight 0.0 --out_dir runs/Curr_cva${IDX}_s1_tt5hz
CKPT1=runs/Curr_cva${IDX}_s1_tt5hz/policy_best.pt

# Stage 2: Contrastive at 10Hz from Stage1
echo "=== $(date) Stage2: Contra@10Hz CVA[$IDX] ==="
python train_rl_fwi.py $BASE --freq 10.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0 --fwi_type contrastive --init_temperature 1.0 --anneal_steps 500 --resume_ckpt $CKPT1 --out_dir runs/Curr_cva${IDX}_s2_contra10hz
CKPT2=runs/Curr_cva${IDX}_s2_contra10hz/policy_best.pt

# Stage 3: Contrastive at 15Hz from Stage2
echo "=== $(date) Stage3: Contra@15Hz CVA[$IDX] ==="
python train_rl_fwi.py $BASE --freq 15.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0 --fwi_type contrastive --init_temperature 1.0 --anneal_steps 500 --resume_ckpt $CKPT2 --out_dir runs/Curr_cva${IDX}_s3_contra15hz

echo "=== DONE CVA[$IDX] ==="
