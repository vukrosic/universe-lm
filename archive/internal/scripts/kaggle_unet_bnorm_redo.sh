#!/usr/bin/env bash
# Corrected bridge-norm redo (earlier run used un-synced code). Waits for the
# token-curve sweep to finish, then runs sigmoid k=6 + bridge_norm at 3M & 10M.
set -eu
REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}; RUNS=${RUNS:-/kaggle/working/unet_runs}; DATA=${DATA:-processed_data/speedrun_40M}
cd "$REPO_DIR"
rm -rf "$RUNS/tiny_unet_bridgenorm_k6_3m" "$RUNS/tiny_unet_bridgenorm_k6_10m"
echo "waiting for token curve to finish..."
while ! grep -q "token curve done" /kaggle/working/tokcurve.out 2>/dev/null; do sleep 30; done
echo "token curve done; waiting for GPUs..."
while true; do u=$(/opt/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|paste -sd+|bc); [ "${u:-1}" -lt 500 ] && break; sleep 20; done
common="--config tiny1m --seed 42 --use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 6 --unet_bridge_norm true --dataset_path $DATA"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u train_llm.py $common --train_tokens 3000000  --log_every 50  --output_dir "$RUNS/tiny_unet_bridgenorm_k6_3m"  > "$RUNS/tiny_unet_bridgenorm_k6_3m.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u train_llm.py $common --train_tokens 10000000 --log_every 100 --output_dir "$RUNS/tiny_unet_bridgenorm_k6_10m" > "$RUNS/tiny_unet_bridgenorm_k6_10m.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "=== bnorm redo done ==="
