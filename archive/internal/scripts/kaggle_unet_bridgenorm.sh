#!/usr/bin/env bash
# Bridge-norm variant: RMSNorm each skip before the gated add.
# sigmoid k=6 + bridge_norm at 3M and 10M. Compare vs existing no-bn runs
# (sigmoid_m15_k6=6.3816 @3M, sigmoid_m15_k6_10m=5.9569 @10M) and ctrl.
set -eu
REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/unet_runs}
DATA=${DATA:-processed_data/speedrun_40M}
cd "$REPO_DIR"
# wait until both GPUs are idle (the 20M pair has finished)
echo "waiting for GPUs to free..."
while true; do
  used=$(/opt/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${used:-1}" -lt 500 ] && break
  sleep 30
done
echo "GPUs free, launching bridge-norm runs"
common="--config tiny1m --seed 42 --use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 6 --unet_bridge_norm true --dataset_path $DATA"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u train_llm.py $common --train_tokens 3000000 --log_every 50 --output_dir "$RUNS/tiny_unet_bridgenorm_k6_3m" > "$RUNS/tiny_unet_bridgenorm_k6_3m.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u train_llm.py $common --train_tokens 10000000 --log_every 100 --output_dir "$RUNS/tiny_unet_bridgenorm_k6_10m" > "$RUNS/tiny_unet_bridgenorm_k6_10m.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "=== bridge-norm done ==="
