#!/bin/bash
# Hyperparameter sweep for latent RL
source /data/shengwz/anaconda3/etc/profile.d/conda.sh
conda activate devito
cd /data/shengwz/swz/RL-seismic-inversion

VAE="runs/vae_200ep/vae_best.pt"
STEPS=200
G=32
PPO=4
SAVE=50
FIDX=50  # CVA[50] as benchmark

for lr in 1e-2 5e-3 2e-3; do
for ent in 0.0 0.02; do
    NAME="lr${lr}_ent${ent}"
    OUT="runs/sweep_${NAME}_cva${FIDX}"
    echo "[sweep] lr=$lr ent=$ent → $OUT"
    mkdir -p "$OUT"
    python -u train_rl_fwi.py \
        --policy_type latent --vae_ckpt "$VAE" \
        --steps $STEPS --group_size $G --ppo_epochs $PPO \
        --model_source cva --cva_file_idx $FIDX --cva_sample_idx 0 \
        --lr $lr --entropy_bonus $ent \
        --save_every $SAVE --out_dir "$OUT" --device cuda:3 \
        > "$OUT/stdout.log" 2>&1
    BEST=$(grep "Best oracle MAE" "$OUT/stdout.log" | tail -1)
    echo "  → $BEST"
done
done

echo "=== SWEEP DONE ==="
for d in runs/sweep_*_cva50; do
    m=$(grep "Best oracle MAE" "$d/stdout.log" 2>/dev/null | tail -1)
    echo "$(basename $d): ${m:-FAILED}"
done
