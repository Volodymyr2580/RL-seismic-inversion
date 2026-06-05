#!/bin/bash
# Multi-model validation: Latent RL + B-spline baseline on CVA test models
# Usage: bash run_multimodel.sh

source /data/shengwz/anaconda3/etc/profile.d/conda.sh
conda activate devito
cd /data/shengwz/swz/RL-seismic-inversion

VAE_CKPT="runs/vae_200ep/vae_best.pt"
TEST_FILES=(52 55 58)
STEPS=200
G=32
PPO=4
SAVE_EVERY=50

echo "=== Multi-model validation ==="
echo "Test files: ${TEST_FILES[@]}"
echo "VAE: $VAE_CKPT"
echo "Steps=$STEPS G=$G ppo_epochs=$PPO"
echo ""

# Run latent RL for each test model
for fidx in "${TEST_FILES[@]}"; do
    OUT="runs/latent_${STEPS}step_cva${fidx}"
    echo "[latent] CVA[$fidx] → $OUT"
    mkdir -p "$OUT"
    python -u train_rl_fwi.py \
        --policy_type latent --vae_ckpt "$VAE_CKPT" \
        --steps $STEPS --group_size $G --ppo_epochs $PPO \
        --model_source cva --cva_file_idx $fidx --cva_sample_idx 0 \
        --save_every $SAVE_EVERY --out_dir "$OUT" --device cuda:0 \
        > "$OUT/stdout.log" 2>&1
    BEST=$(grep "Best oracle MAE" "$OUT/stdout.log" | tail -1)
    echo "  $BEST"
done

# Run B-spline baseline for each test model
for fidx in "${TEST_FILES[@]}"; do
    OUT="runs/bspline_${STEPS}step_cva${fidx}"
    echo "[bspline] CVA[$fidx] → $OUT"
    mkdir -p "$OUT"
    python -u train_rl_fwi.py \
        --policy_type mean \
        --steps $STEPS --group_size $G --ppo_epochs $PPO \
        --model_source cva --cva_file_idx $fidx --cva_sample_idx 0 \
        --save_every $SAVE_EVERY --out_dir "$OUT" --device cuda:0 \
        > "$OUT/stdout.log" 2>&1
    BEST=$(grep "Best oracle MAE" "$OUT/stdout.log" | tail -1)
    echo "  $BEST"
done

echo ""
echo "=== Done ==="
grep -H "Best oracle MAE" runs/latent_${STEPS}step_cva5*/stdout.log runs/bspline_${STEPS}step_cva5*/stdout.log 2>/dev/null
