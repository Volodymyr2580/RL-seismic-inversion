#!/bin/bash
# Band-filtered curriculum: same p_data, progressive lowpass cutoff
# Stage1: TT@0-5Hz → Stage2: Contra@0-10Hz → Stage3: Contra@full
GPU=$1; IDX=$2
LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
cd /data/shengwz/swz/RL-seismic-inversion

BASE="--policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx $IDX --geometry transmission --reward_l1_weight 0.0 --reward_prior_weight 0.0 --best_criterion l2 --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 --entropy_bonus 0.02 --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100 --seed 42"

# Stage 1: TT-only, reward computed on 0-5Hz band
echo "=== $(date) Stage1: TT@0-5Hz CVA[$IDX] ==="
python train_rl_fwi.py $BASE --init_temperature 2.0 --final_temperature 0.1 --anneal_steps 1000 \
  --reward_tt_weight 1.0 --reward_tt_log --reward_l2_weight 0.0 \
  --freq_band 0-5 --out_dir runs/Band_cva${IDX}_s1_tt_0-5
CKPT1=runs/Band_cva${IDX}_s1_tt_0-5/policy_best.pt

# Stage 2: Contrastive, reward on 0-10Hz band, from Stage1 checkpoint
echo "=== $(date) Stage2: Contra@0-10Hz CVA[$IDX] ==="
python train_rl_fwi.py $BASE --init_temperature 1.0 --final_temperature 0.1 --anneal_steps 500 \
  --reward_l2_weight 1.0 --reward_tt_weight 0.0 --fwi_type contrastive \
  --freq_band 0-10 --resume_ckpt $CKPT1 --out_dir runs/Band_cva${IDX}_s2_contra_0-10
CKPT2=runs/Band_cva${IDX}_s2_contra_0-10/policy_best.pt

# Stage 3: Contrastive, full bandwidth, from Stage2 checkpoint
echo "=== $(date) Stage3: Contra@full CVA[$IDX] ==="
python train_rl_fwi.py $BASE --init_temperature 1.0 --final_temperature 0.1 --anneal_steps 500 \
  --reward_l2_weight 1.0 --reward_tt_weight 0.0 --fwi_type contrastive \
  --resume_ckpt $CKPT2 --out_dir runs/Band_cva${IDX}_s3_contra_full

echo "=== DONE CVA[$IDX] ==="
