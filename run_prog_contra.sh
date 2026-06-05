#!/bin/bash
# Progressive TT → Contrastive
GPU=$1
LDP=/data/shengwz/.local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$LDP/cublas/lib:$LDP/cuda_cupti/lib:$LDP/cuda_nvrtc/lib:$LDP/cuda_runtime/lib:$LDP/cudnn/lib:$LDP/cufft/lib:$LDP/curand/lib:$LDP/cusolver/lib:$LDP/cusparse/lib:$LDP/nvtx/lib
export CUDA_VISIBLE_DEVICES=$GPU
cd /data/shengwz/swz/RL-seismic-inversion

# (model_idx, tt_checkpoint_dir)
for pair in "18 B_cva18" "50 phase4_gauss_logtt_s50" "10 B_cva10" "6 B_cva6" "8 M8_cva8" "5 M5_cva5"; do
  idx=$(echo $pair | cut -d' ' -f1)
  tt_dir=$(echo $pair | cut -d' ' -f2)
  echo "=== $(date) Prog TT→Contrastive CVA[$idx] GPU=$GPU ==="
  python train_rl_fwi.py \
    --policy_type gaussian --model_source smooth --smooth_root data/smooth_models_v2 --cva_file_idx $idx \
    --geometry transmission --reward_l2_weight 1.0 --reward_l1_weight 0.0 --reward_tt_weight 0.0 --reward_prior_weight 0.0 \
    --fwi_type contrastive --best_criterion l2 \
    --steps 5000 --group_size 32 --ppo_epochs 4 --lr 5e-3 \
    --init_temperature 1.0 --final_temperature 0.1 --anneal_steps 500 \
    --entropy_bonus 0.02 \
    --resume_ckpt runs/${tt_dir}/policy_best.pt \
    --out_dir runs/ProgContra_cva${idx} \
    --device cuda --early_stop_patience 500 --early_stop_window 100 --save_every 100 --seed 42
done
echo "=== DONE GPU=$GPU ==="
