#!/usr/bin/env bash
# Phase7 reward-function suite:
# 3 models x 3 deterministic initial models x 8 reward settings.
#
# Example on the server:
#   bash run_phase7_reward_suite.sh
#   STEPS=1200 GROUP_SIZE=32 GPU_LIST=0,1,2,3 bash run_phase7_reward_suite.sh
#
# Optional filters:
#   MODELS=marmousi,salt_A_90x270 REWARDS=l1l2,ncc_maxlag INITS=far_uniform bash run_phase7_reward_suite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$SCRIPT_DIR}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "/data/shengwz/anaconda3/envs/devito/bin/python" ]]; then
    PY="/data/shengwz/anaconda3/envs/devito/bin/python"
  else
    PY="python"
  fi
fi

if [[ -d "/data/shengwz/anaconda3/envs/devito/lib" ]]; then
  export LD_LIBRARY_PATH="/data/shengwz/anaconda3/envs/devito/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$PROJ"

STEPS="${STEPS:-1000}"
GROUP_SIZE="${GROUP_SIZE:-32}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
LR="${LR:-5e-3}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-500}"
EARLY_STOP_WINDOW="${EARLY_STOP_WINDOW:-100}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NT="${NT:-1000}"
N_SHOTS="${N_SHOTS:-7}"
DX="${DX:-10.0}"
DT="${DT:-0.001}"
FREQ="${FREQ:-15.0}"
PML_WIDTH="${PML_WIDTH:-40}"
RUN_ROOT="${RUN_ROOT:-runs/phase7}"
INIT_ROOT="${INIT_ROOT:-outputs/phase7_sensitivity/initial_models}"
REPORT_DIR="${REPORT_DIR:-reports/phase7_reward_suite}"

MODEL_FILTER="${MODELS:-marmousi,salt_A_90x270,synthetic}"
INIT_FILTER="${INITS:-far_uniform,medium_gradient,near_blur}"
REWARD_FILTER="${REWARDS:-tt_only,l1l2,wasserstein_w1,wasserstein_w2,ncc_zero,ncc_maxlag,envelope_ncc,awi}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"
if [[ "$NGPU" -lt 1 ]]; then
  echo "[error] GPU_LIST is empty"
  exit 1
fi

contains_csv() {
  local csv="$1"
  local item="$2"
  [[ ",$csv," == *",$item,"* ]]
}

model_args() {
  local model="$1"
  case "$model" in
    marmousi)
      MODEL_PATH="outputs/phase7_sensitivity/marmousi_nx210_nz70_ctrl30x10/v_true.npy"
      NX=210; NZ=70; NX_CTRL=30; NZ_CTRL=10; NREC=210; VMIN=1000; VMAX=5600
      ;;
    salt_A_90x270)
      MODEL_PATH="outputs/phase7_sensitivity/salt_nx270_nz90_ctrl36x12/v_true.npy"
      NX=270; NZ=90; NX_CTRL=36; NZ_CTRL=12; NREC=270; VMIN=1000; VMAX=5600
      ;;
    synthetic)
      MODEL_PATH="outputs/phase7_sensitivity/synthetic_nx210_nz70_ctrl30x10/v_true.npy"
      NX=210; NZ=70; NX_CTRL=30; NZ_CTRL=10; NREC=210; VMIN=1000; VMAX=5600
      ;;
    *)
      echo "[error] unknown model: $model"
      exit 1
      ;;
  esac
}

reward_args() {
  local reward="$1"
  REWARD_ARGS=()
  case "$reward" in
    tt_only)
      REWARD_ARGS=(--fwi_type l2 --reward_l1_weight 0.0 --reward_l2_weight 0.0 --reward_tt_weight 1.0 --reward_tt_log)
      ;;
    l1l2)
      REWARD_ARGS=(--fwi_type l2 --reward_l1_weight 1.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0)
      ;;
    wasserstein_w1)
      REWARD_ARGS=(--fwi_type wasserstein --wasserstein_normalize abs --reward_l1_weight 0.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0)
      ;;
    wasserstein_w2)
      REWARD_ARGS=(--fwi_type wasserstein_w2 --reward_l1_weight 0.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0)
      ;;
    ncc_zero|ncc_maxlag|envelope_ncc|awi)
      REWARD_ARGS=(--fwi_type "$reward" --reward_l1_weight 0.0 --reward_l2_weight 1.0 --reward_tt_weight 0.0)
      ;;
    *)
      echo "[error] unknown reward: $reward"
      exit 1
      ;;
  esac
}

echo "[phase7] python: $PY"
echo "[phase7] project: $PROJ"
echo "[phase7] GPUs: $GPU_LIST"
echo "[phase7] steps=$STEPS group_size=$GROUP_SIZE ppo_epochs=$PPO_EPOCHS"

"$PY" scripts/phase7_prepare_initial_models.py --out_root "$INIT_ROOT"

MODELS_ALL=(marmousi salt_A_90x270 synthetic)
INITS_ALL=(far_uniform medium_gradient near_blur)
REWARDS_ALL=(tt_only l1l2 wasserstein_w1 wasserstein_w2 ncc_zero ncc_maxlag envelope_ncc awi)

JOBS=()
for model in "${MODELS_ALL[@]}"; do
  contains_csv "$MODEL_FILTER" "$model" || continue
  for init in "${INITS_ALL[@]}"; do
    contains_csv "$INIT_FILTER" "$init" || continue
    for reward in "${REWARDS_ALL[@]}"; do
      contains_csv "$REWARD_FILTER" "$reward" || continue
      JOBS+=("$model|$init|$reward")
    done
  done
done

TOTAL="${#JOBS[@]}"
if [[ "$TOTAL" -lt 1 ]]; then
  echo "[error] no jobs selected"
  exit 1
fi
echo "[phase7] selected jobs: $TOTAL"

mkdir -p "$RUN_ROOT/logs"

run_job() {
  local gpu="$1"
  local model="$2"
  local init="$3"
  local reward="$4"

  model_args "$model"
  reward_args "$reward"

  local init_path="$INIT_ROOT/$model/$init.npy"
  local run_dir="$RUN_ROOT/$model/$init/$reward"
  local log_path="$RUN_ROOT/logs/${model}_${init}_${reward}.log"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[error] missing true model: $MODEL_PATH"
    exit 1
  fi
  if [[ ! -f "$init_path" ]]; then
    echo "[error] missing init model: $init_path"
    exit 1
  fi

  mkdir -p "$run_dir"

  if [[ -f "$run_dir/policy_final.pt" && -f "$run_dir/final_velocity.npy" ]]; then
    echo "[skip][gpu $gpu] $model/$init/$reward already complete"
    return 0
  fi

  echo "[run][gpu $gpu] $model/$init/$reward"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train_rl_fwi.py \
    --model_source npy \
    --model_path "$MODEL_PATH" \
    --init_velocity_path "$init_path" \
    --policy_type mean \
    --geometry transmission \
    --source_depth bottom \
    --receiver_depth top \
    --nx_model "$NX" \
    --nz_model "$NZ" \
    --nx_ctrl "$NX_CTRL" \
    --nz_ctrl "$NZ_CTRL" \
    --v_min "$VMIN" \
    --v_max "$VMAX" \
    --n_shots "$N_SHOTS" \
    --n_receivers "$NREC" \
    --nt "$NT" \
    --dx "$DX" \
    --dt "$DT" \
    --freq "$FREQ" \
    --pml_width "$PML_WIDTH" \
    --group_size "$GROUP_SIZE" \
    --steps "$STEPS" \
    --ppo_epochs "$PPO_EPOCHS" \
    --lr "$LR" \
    --init_temperature 2.0 \
    --final_temperature 0.1 \
    --anneal_steps "$STEPS" \
    --entropy_bonus 0.02 \
    --reward_prior_weight 0.0 \
    --best_criterion l2 \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --early_stop_window "$EARLY_STOP_WINDOW" \
    --save_every "$SAVE_EVERY" \
    --device cuda \
    --seed 202607 \
    --out_dir "$run_dir" \
    "${REWARD_ARGS[@]}" \
    > "$log_path" 2>&1

  echo "[viz][gpu $gpu] $model/$init/$reward"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/phase7_visualize_run.py "$run_dir" --device cuda >> "$log_path" 2>&1; then
    echo "[warn] visualization failed for $model/$init/$reward; see $log_path"
  fi
}

worker() {
  local worker_id="$1"
  local gpu="${GPUS[$worker_id]}"
  local idx="$worker_id"
  while [[ "$idx" -lt "$TOTAL" ]]; do
    IFS='|' read -r model init reward <<< "${JOBS[$idx]}"
    run_job "$gpu" "$model" "$init" "$reward"
    idx=$((idx + NGPU))
  done
}

PIDS=()
for wid in $(seq 0 $((NGPU - 1))); do
  worker "$wid" &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[phase7] some jobs failed; building partial report"
else
  echo "[phase7] all jobs finished"
fi

"$PY" scripts/phase7_build_report.py --runs_root "$RUN_ROOT" --report_dir "$REPORT_DIR"
echo "[phase7] report: $REPORT_DIR/index.html"

exit "$FAIL"
